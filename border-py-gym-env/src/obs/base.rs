use crate::util::pyobj_to_arrayd;
use border_core::Obs;
use ndarray::{ArrayD, IxDyn};
use num_traits::cast::AsPrimitive;
use numpy::Element;
use pyo3::PyObject;
use std::fmt::Debug;
use std::marker::PhantomData;
#[cfg(feature = "tch")]
use {std::convert::TryFrom, tch::Tensor};

/// Observation represented by an [ndarray::ArrayD].
///
/// `S` is the shape of an observation, except for batch and process dimensions.
/// `T` is the dtype of ndarray in the Python gym environment.
/// For some reason, the dtype of observations in Python gym environments seems to
/// vary, f32 or f64. To get observations in Rust side, the dtype is specified as a
/// type parameter, instead of checking the dtype of Python array at runtime.
#[deprecated]
#[derive(Clone, Debug)]
pub struct GymObs<T1, T2>
where
    T1: Element + Debug,
    T2: 'static + Copy,
{
    pub obs: ArrayD<T2>,
    pub(crate) phantom: PhantomData<T1>,
}

impl<T1, T2> From<ArrayD<T2>> for GymObs<T1, T2>
where
    T1: Element + Debug,
    T2: 'static + Copy,
{
    fn from(obs: ArrayD<T2>) -> Self {
        Self {
            obs,
            phantom: PhantomData,
        }
    }
}

impl<T1, T2> Obs for GymObs<T1, T2>
where
    T1: Debug + Element,
    T2: 'static + Copy + Debug + num_traits::Zero,
{
    fn dummy(_n_procs: usize) -> Self {
        // let shape = &mut S::shape().to_vec();
        // shape.insert(0, n_procs as _);
        // trace!("Shape of TchPyGymEnvObs: {:?}", shape);
        let shape = vec![0];
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.obs.shape()[0]
    }
}

/// Convert numpy array of Python into [`GymObs`].
impl<T1, T2> From<PyObject> for GymObs<T1, T2>
where
    T1: Element + AsPrimitive<T2> + std::fmt::Debug,
    T2: 'static + Copy,
{
    fn from(obs: PyObject) -> Self {
        Self {
            obs: pyobj_to_arrayd::<T1, T2>(obs),
            phantom: PhantomData,
        }
    }
}

// #[cfg(feature = "tch")]
// impl<S, T1, T2> From<PyGymEnvObs<S, T1, T2>> for Tensor
// where
//     S: Shape,
//     T1: Element + Debug,
//     T2: 'static + Copy,
// {
//     fn from(obs: PyGymEnvObs<S, T1, T2>) -> Tensor {
//         let tmp = &obs.obs;
//         Tensor::try_from(tmp).unwrap()
//         // Tensor::try_from(&obs.obs).unwrap()
//     }
// }

#[cfg(feature = "tch")]
impl<T1> From<GymObs<T1, f32>> for Tensor
where
    T1: Element + Debug,
{
    fn from(obs: GymObs<T1, f32>) -> Tensor {
        let tmp = &obs.obs;
        Tensor::try_from(tmp).unwrap()
        // Tensor::try_from(&obs.obs).unwrap()
    }
}

#[cfg(feature = "tch")]
impl<T1> From<GymObs<T1, u8>> for Tensor
where
    T1: Element + Debug,
{
    fn from(obs: GymObs<T1, u8>) -> Tensor {
        let tmp = &obs.obs;
        Tensor::try_from(tmp).unwrap()
        // Tensor::try_from(&obs.obs).unwrap()
    }
}
