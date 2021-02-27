use std::{fmt::Debug, iter::FromIterator};
use std::default::Default;
use std::marker::PhantomData;
use log::trace;
use num_traits::cast::AsPrimitive;
use pyo3::{Py, PyObject, types::PyList};
use ndarray::{ArrayD, Axis, IxDyn, stack};
use numpy::{Element, PyArrayDyn};

use crate::{
    core::{Obs, record::{Record, RecordValue}},
    env::py_gym_env::{Shape, PyGymEnvObsFilter},
};

#[derive(Debug)]
struct FrameStackFilter {
    data: ArrayD<f32>,
    nprocs: i64,
    nstack: i64,
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

// impl FrameStack {
//     fn new(nprocs: i64, nstack: i64) -> FrameStack {
//         FrameStack {
//             data: Tensor::zeros(&[nprocs, nstack, 84, 84], FLOAT_CPU),
//             nprocs,
//             nstack,
//         }
//     }

//     fn update<'a>(&'a mut self, img: &Tensor, masks: Option<&Tensor>) -> &'a Tensor {
//         if let Some(masks) = masks {
//             self.data *= masks.view([self.nprocs, 1, 1, 1])
//         };
//         let slice = |i| self.data.narrow(1, i, 1);
//         for i in 1..self.nstack {
//             slice(i - 1).copy_(&slice(i))
//         }
//         slice(self.nstack - 1).copy_(img);
//         &self.data
//     }
// }
