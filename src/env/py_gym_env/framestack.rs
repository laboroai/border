//! An observation filter with staking observations (frames).
use std::{fmt::Debug, marker::PhantomData};
use num_traits::cast::AsPrimitive;
use pyo3::{Py, PyAny, PyObject, types::PyList};
use ndarray::{ArrayD, Axis, IxDyn, SliceInfo, SliceOrIndex, stack};
use numpy::{Element, PyArrayDyn};

use crate::{
    core::record::Record,
    env::py_gym_env::{Shape, obs::PyGymEnvObs, PyGymEnvObsFilter},
};

/// An observation filter with stacking sequence of original observations.
///
/// This filter only supports vectorized environments ([`crate::env::py_gym_env::PyVecGymEnv`]).
/// The first element of the shape `S` denotes the number of stacks (`n_stack`) and the following elements
/// denote the shape of the partial observation, which is the observation of each environment
/// in the vectorized environment.
#[derive(Debug)]
pub struct FrameStackFilter<S, T1, T2> where
    T2: 'static + Copy
{
    buffer: Vec<ArrayD<T2>>,
    n_procs: i64,
    n_stack: i64,
    vectorized: bool,
    phantom: PhantomData<(S, T1, T2)>
}

impl<S, T1, T2> FrameStackFilter<S, T1, T2> where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero
{
    /// Constructs an observation filter for single process environments.
    pub fn new(n_stack: i64) -> FrameStackFilter<S, T1, T2> {
        FrameStackFilter {
            buffer: (0..1).map(|_| {ArrayD::<T2>::zeros(S::shape())}).collect(),
            n_procs: 1,
            n_stack,
            vectorized: false,
            phantom: PhantomData,
        }
    }

    /// Constructs an observation filter for vectorized environments.
    pub fn vectorized(n_procs: i64, n_stack: i64) -> FrameStackFilter<S, T1, T2> {
        FrameStackFilter {
            buffer: (0..n_procs).map(|_| {ArrayD::<T2>::zeros(S::shape())}).collect(),
            n_procs,
            n_stack,
            vectorized: true, // note: n_procs can be 1 when vectorized is true
            phantom: PhantomData,
        }
    }

    /// Create slice for a dynamic array.
    /// See https://github.com/rust-ndarray/ndarray/issues/501
    fn s(j: usize) -> SliceInfo<Vec<SliceOrIndex>, IxDyn> {
        let slicer = vec![SliceOrIndex::Index(j as isize)].into_iter()
            .chain(
                (1..S::shape().len()).map(|_| SliceOrIndex::Slice {
                    start: 0,
                    end: None,
                    step: 1
                }).into_iter()                
            )
            .collect();
        SliceInfo::new(slicer).unwrap()
    }

    /// Update the buffer of the stacked observations.
    fn update_buffer(&mut self, i: i64, obs: &ArrayD<T2>) {
        if let Some(arr) = self.buffer.get_mut(i as usize) {
            // Shift stacks frame(j) <- frame(j - 1) for j=1,..,(n_stack - 1)
            for j in (1..self.n_stack as usize).rev() {
                let (mut dst, src) = arr.multi_slice_mut(
                    (Self::s(j).as_ref(), Self::s(j - 1).as_ref())
                );
                dst.assign(&src);
            }
            arr.slice_mut(Self::s(0).as_ref()).assign(obs)
        }
        else {
            unimplemented!()
        }
    }

    /// Fill the buffer, invoked when resetting
    fn fill_buffer(&mut self, i: i64, obs: &ArrayD<T2>) {
        if let Some(arr) = self.buffer.get_mut(i as usize) {
            for j in (0..self.n_stack as usize).rev() {
                let mut dst = arr.slice_mut(Self::s(j).as_ref());
                dst.assign(&obs);
            }
        }
        else {
            unimplemented!();
        }
    }

    /// Get ndarray from pyobj
    fn get_ndarray(o: &PyAny) -> ArrayD<T2> {
        debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
        let o: &PyArrayDyn<T1> = o.extract().unwrap();
        let o = o.to_owned_array();
        let o = o.mapv(|elem| elem.as_());
        debug_assert_eq!(o.shape()[..], S::shape()[1..]);
        o
    }
}

impl<S, T1, T2> PyGymEnvObsFilter<PyGymEnvObs<S, T1, T2>> for FrameStackFilter<S, T1, T2> where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + Debug + num_traits::Zero
{
    fn filt(&mut self, obs: PyObject) -> (PyGymEnvObs<S, T1, T2>, Record) {
        if self.vectorized {
            // Processes the input observation to update `self.buffer`
            pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");

                let obs: Py<PyList> = obs.extract(py).unwrap();

                for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
                    let o = Self::get_ndarray(o);
                    self.update_buffer(i, &o);
                }
            });

            // Returned values
            let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
            let obs = PyGymEnvObs::<S, T1, T2> {
                obs: stack(Axis(0), array_views.as_slice()).unwrap(),
                phantom: PhantomData
            };

            // TODO: add contents in the record
            let record = Record::empty();

            (obs, record)
        }
        else {
            // Update the buffer with obs
            pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "ndarray");
                let o = Self::get_ndarray(obs.as_ref(py));
                self.update_buffer(0, &o);
            });

            // Returns stacked observation in the buffer
            let obs = PyGymEnvObs::<S, T1, T2> {
                obs: self.buffer[0].clone().insert_axis(Axis(0)),
                phantom: PhantomData
            };

            // TODO: add contents in the record
            let record = Record::empty();

            (obs, record)
        }
    }

    fn reset(&mut self, obs: PyObject) -> PyGymEnvObs::<S, T1, T2> {
        if self.vectorized {
            pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");
    
                let obs: Py<PyList> = obs.extract(py).unwrap();
    
                for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
                    if o.get_type().name().unwrap() != "NoneType" {
                        let o = Self::get_ndarray(o);
                        self.fill_buffer(i, &o);
                    }
                }
            });
    
            // Returned values
            let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
            PyGymEnvObs::<S, T1, T2> {
                obs: stack(Axis(0), array_views.as_slice()).unwrap(),
                phantom: PhantomData
            }
        }
        else {
            // Update the buffer if obs is not None, otherwise do nothing
            pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() != "NoneType" {
                    debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "ndarray");
                    let o = Self::get_ndarray(obs.as_ref(py));
                    self.fill_buffer(0, &o);
                }
            });

            // Returns stacked observation in the buffer
            PyGymEnvObs::<S, T1, T2> {
                obs: self.buffer[0].clone().insert_axis(Axis(0)),
                phantom: PhantomData
            }
        }
    }
}
