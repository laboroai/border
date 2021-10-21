//! An observation filter with stacking observations (frames).
use super::PyGymEnvObs;
use crate::PyGymEnvObsFilter;
use border_core::Shape;
use border_core::{
    record::{Record, RecordValue},
    Obs,
};
use ndarray::{stack, ArrayD, Axis, IxDyn, SliceInfo, SliceInfoElem}; //, SliceOrIndex};
use num_traits::cast::AsPrimitive;
use numpy::{Element, PyArrayDyn};
use pyo3::{types::PyList, Py, PyAny, PyObject};
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, fmt::Debug, marker::PhantomData};

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [FrameStackFilter].
pub struct FrameStackFilterConfig {
    n_procs: i64,
    n_stack: i64,
    vectorized: bool,
}

impl Default for FrameStackFilterConfig {
    fn default() -> Self {
        Self {
            n_procs: 1,
            n_stack: 4,
            vectorized: false,
        }
    }
}

/// An observation filter with stacking sequence of original observations.
///
/// This filter supports vectorized environments ([crate::PyVecGymEnv]).
/// The first element of the shape `S` denotes the number of stacks (`n_stack`) and the following elements
/// denote the shape of the partial observation, which is the observation of each environment
/// in the vectorized environment.
#[derive(Debug)]
pub struct FrameStackFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero,
    U: Obs + From<PyGymEnvObs<S, T1, T2>>,
{
    buffer: Vec<ArrayD<T2>>,
    n_procs: i64,
    n_stack: i64,
    vectorized: bool,
    phantom: PhantomData<(S, T1, U)>,
}

impl<S, T1, T2, U> FrameStackFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero,
    U: Obs + From<PyGymEnvObs<S, T1, T2>>,
{
    /// Returns the default configuration.
    pub fn default_config() -> FrameStackFilterConfig {
        FrameStackFilterConfig::default()
    }

    /// Create slice for a dynamic array: equivalent to arr[j:(j+1), ::] in numpy.
    ///
    /// See https://github.com/rust-ndarray/ndarray/issues/501
    fn s(j: usize) -> Vec<SliceInfoElem> {
        // The first index of S::shape() corresponds to stacking dimension,
        // specific index.
        let mut slicer = vec![SliceInfoElem::Index(j as isize)];

        // For remaining dimensions, all elements will be taken.
        let n = S::shape().len() - 1;
        let (start, end, step) = (0, None, 1);

        slicer.extend(vec![SliceInfoElem::Slice { start, end, step }; n]);
        slicer
    }

    /// Update the buffer of the stacked observations.
    fn update_buffer(&mut self, i: i64, obs: &ArrayD<T2>) {
        if let Some(arr) = self.buffer.get_mut(i as usize) {
            // Shift stacks frame(j) <- frame(j - 1) for j=1,..,(n_stack - 1)
            for j in (1..self.n_stack as usize).rev() {
                let dst_slice = &Self::s(j)[..];
                let src_slice = &Self::s(j - 1)[..];
                let (mut dst, src) = arr.multi_slice_mut((dst_slice, src_slice));
                dst.assign(&src);
            }
            arr.slice_mut(&Self::s(0)[..]).assign(obs)
        } else {
            unimplemented!()
        }
    }

    /// Fill the buffer, invoked when resetting
    fn fill_buffer(&mut self, i: i64, obs: &ArrayD<T2>) {
        if let Some(arr) = self.buffer.get_mut(i as usize) {
            for j in (0..self.n_stack as usize).rev() {
                let mut dst = arr.slice_mut(&Self::s(j)[..]);
                dst.assign(&obs);
            }
        } else {
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

impl<S, T1, T2, U> PyGymEnvObsFilter<U> for FrameStackFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero + Into<f32>,
    U: Obs + From<PyGymEnvObs<S, T1, T2>>,
{
    type Config = FrameStackFilterConfig;

    fn build(config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(FrameStackFilter {
            buffer: (0..1).map(|_| ArrayD::<T2>::zeros(S::shape())).collect(),
            n_procs: config.n_procs,
            n_stack: config.n_stack,
            vectorized: config.vectorized,
            phantom: PhantomData,
        })
    }

    fn filt(&mut self, obs: PyObject) -> (U, Record) {
        if self.vectorized {
            unimplemented!();
            // // Processes the input observation to update `self.buffer`
            // pyo3::Python::with_gil(|py| {
            //     debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");

            //     let obs: Py<PyList> = obs.extract(py).unwrap();

            //     for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
            //         let o = Self::get_ndarray(o);
            //         self.update_buffer(i, &o);
            //     }
            // });

            // // Returned values
            // let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
            // let obs = PyGymEnvObs::from(stack(Axis(0), array_views.as_slice()).unwrap());
            // let obs = U::from(obs);

            // // TODO: add contents in the record
            // let record = Record::empty();

            // (obs, record)
        } else {
            // Update the buffer with obs
            pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "ndarray");
                let o = Self::get_ndarray(obs.as_ref(py));
                self.update_buffer(0, &o);
            });

            // Returns stacked observation in the buffer
            // img.shape() = [1, 4, 1, 84, 84]
            // [batch_size, n_stack, color_ch, width, height]
            let img = self.buffer[0].clone().insert_axis(Axis(0));
            let data = img.iter().map(|&e| e.into()).collect::<Vec<_>>();
            let shape = [img.shape()[3] * self.n_stack as usize, img.shape()[4]];

            let obs = PyGymEnvObs::from(img);
            let obs = U::from(obs);

            // TODO: add contents in the record
            let mut record = Record::empty();
            record.insert("frame_stack_filter_out", RecordValue::Array2(data, shape));

            (obs, record)
        }
    }

    fn reset(&mut self, obs: PyObject) -> U {
        if self.vectorized {
            unimplemented!();
            // pyo3::Python::with_gil(|py| {
            //     debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");

            //     let obs: Py<PyList> = obs.extract(py).unwrap();

            //     for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
            //         if o.get_type().name().unwrap() != "NoneType" {
            //             let o = Self::get_ndarray(o);
            //             self.fill_buffer(i, &o);
            //         }
            //     }
            // });

            // // Returned values
            // let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
            // O::from(stack(Axis(0), array_views.as_slice()).unwrap())
        } else {
            // Update the buffer if obs is not None, otherwise do nothing
            pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() != "NoneType" {
                    debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "ndarray");
                    let o = Self::get_ndarray(obs.as_ref(py));
                    self.fill_buffer(0, &o);
                }
            });

            // Returns stacked observation in the buffer
            let frames = self.buffer[0].clone().insert_axis(Axis(0));
            U::from(PyGymEnvObs::from(frames))
        }
    }
}
