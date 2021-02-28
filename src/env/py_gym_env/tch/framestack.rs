use std::{fmt::Debug, iter::FromIterator, default::Default, marker::PhantomData};
use log::trace;
use num_traits::cast::AsPrimitive;
use pyo3::{Py, PyAny, PyObject, types::PyList};
use ndarray::{ArrayD, Axis, IxDyn, Slice, SliceInfo, SliceOrIndex, stack};
use numpy::{Element, PyArrayDyn};

use crate::{
    core::{Obs, record::{Record, RecordValue}},
    env::py_gym_env::{Shape, obs::PyGymEnvObs, PyGymEnvObsFilter},
};

/// An observation filter with stacking sequence of original observations.
///
/// This filter only supports vectorized environments ([`super::super::PyVecGymEnv`]).
/// The first element of the shape `S` denotes the number of stacks (`n_stack`) and the following elements
/// denote the shape of the partial observation, which is the observation of each environment
/// in the vectorized environment.
#[derive(Debug)]
pub struct FrameStackFilter<S, T> {
    buffer: Vec<ArrayD<f32>>,
    n_procs: i64,
    n_stack: i64,
    phantom: PhantomData<(S, T)>
}

/// Convert `PyObject` to [`ndarray::ArrayD`].
///
/// This function is used with [`FrameStack`].
/// This function supports only vectorized environments ([`super::super::vec::PyVecGymEnv`]).
pub fn pyobj_to_arrayd<S, T>(obs: PyObject) -> ArrayD<f32> where
    S: Shape,
    T: Element + AsPrimitive<f32>,
{
    unimplemented!();
}

impl<S, T> FrameStackFilter<S, T> where
    S: Shape
{
    // fn new(nprocs: i64, nstack: i64) -> FrameStack {
    //     FrameStack {
    //         data: Tensor::zeros(&[nprocs, nstack, 84, 84], FLOAT_CPU),
    //         nprocs,
    //         nstack,
    //     }
    // }

    // fn update<'a>(&'a mut self, img: &Tensor, masks: Option<&Tensor>) -> &'a Tensor {
    //     if let Some(masks) = masks {
    //         self.data *= masks.view([self.nprocs, 1, 1, 1])
    //     };
    //     let slice = |i| self.data.narrow(1, i, 1);
    //     for i in 1..self.nstack {
    //         slice(i - 1).copy_(&slice(i))
    //     }
    //     slice(self.nstack - 1).copy_(img);
    //     &self.data
    // }

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
    fn update_buffer(&mut self, i: i64, obs: &ArrayD<f32>) {
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
}

impl<S, T> PyGymEnvObsFilter<PyGymEnvObs<S, T>> for FrameStackFilter<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero + AsPrimitive<f32>,
{
    fn filt(&mut self, obs: PyObject) -> (PyGymEnvObs<S, T>, Record) {
        // Processes the input observation to update `self.buffer`
        pyo3::Python::with_gil(|py| {
            debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");
            let obs: Py<PyList> = obs.extract(py).unwrap();
            for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
                if o.get_type().name().unwrap() != "NoneType" {
                    debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
                    let o: &PyArrayDyn<T> = o.extract().unwrap();
                    let o = o.to_owned_array();
                    let o = o.mapv(|elem| elem.as_());
                    debug_assert_eq!(o.shape(), S::shape());
                    self.update_buffer(i, &o);
                }
            }
        });

        // Returned values
        let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
        let obs = PyGymEnvObs::<S, T> {
            obs: stack(Axis(0), array_views.as_slice()).unwrap(),
            phantom: PhantomData
        };

        // TODO: add contents in the record
        let record = Record::empty();

        (obs, record)
    }
}