use std::fmt::Debug;
use std::marker::PhantomData;
use log::trace;
use num_traits::cast::AsPrimitive;
use pyo3::{PyObject};
use ndarray::{ArrayD, Axis, IxDyn};
use numpy::{PyArrayDyn, Element};
use tch::Tensor;
use crate::core::Obs;
use crate::agents::tch::{Shape, TchBuffer, util::try_from, util::concat_slices};
use crate::py_gym_env::ObsFilter;

// use numpy::num_tra

fn any(is_done: &[f32]) -> bool {
    is_done.iter().fold(0, |x, v| x + *v as i32) > 0
}

fn pyobj_to_arrayd<S, T>(obs: PyObject) -> ArrayD<f32> where
    S: Shape,
    T: Element + AsPrimitive<f32>,
{
    pyo3::Python::with_gil(|py| {
        let obs: &PyArrayDyn<T> = obs.extract(py).unwrap();
        let obs = obs.to_owned_array();
        // let obs = obs.mapv(|elem| elem as f32);
        let obs = obs.mapv(|elem| elem.as_());
        let obs = {
            if obs.shape().len() == S::shape().len() + 1 {
                // In this case obs has a dimension for n_procs
                obs
            }
            else if obs.shape().len() == S::shape().len() {
                // add dimension for n_procs
                obs.insert_axis(Axis(0))
            }
            else {
                panic!();
            }
        };
        obs
    })
}

pub struct TchPyGymEnvObsRawFilter<S, T> {
    phantom: PhantomData<(S, T)>
}

impl <S, T>TchPyGymEnvObsRawFilter<S, T> where
    S: Shape,
    T: Element,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<S, T> ObsFilter<TchPyGymEnvObs<S, T>> for TchPyGymEnvObsRawFilter<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero + AsPrimitive<f32>,
{
    fn filt(&self, obs: PyObject) -> TchPyGymEnvObs<S, T> {
        TchPyGymEnvObs {
            obs: pyobj_to_arrayd::<S, T>(obs),
            phantom: PhantomData,
        }
    }
}

/// Represents observation.
#[derive(Clone, Debug)]
pub struct TchPyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug,
{
    obs: ArrayD<f32>,
    phantom: PhantomData<(S, T)>
}

impl<S, T> Obs for TchPyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero,
{
    fn zero(n_procs: usize) -> Self {
        let shape = &mut S::shape().to_vec();
        shape.insert(0, n_procs as _);
        trace!("Shape of TchPyGymEnvObs: {:?}", shape);
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData
        }
    }

    fn merge(mut self, obs_reset: Self, is_done: &[f32]) -> Self {
        if any(is_done) {
            for (i, is_done_i) in is_done.iter().enumerate() {
                if *is_done_i != 0.0 as f32 {
                    self.obs.index_axis_mut(Axis(0), i)
                        .assign(&obs_reset.obs.index_axis(Axis(0), i));
                }
            }
        };
        self
    }
}

// impl<S: Shape> From<PyObject> for TchPyGymEnvObs<S, f32>
// {
//     fn from(obs: PyObject) -> Self {
//         pyo3::Python::with_gil(|py| {
//             // let obs: &PyArrayDyn<f64> = obs.extract(py).unwrap();
//             let obs: &PyArrayDyn<f32> = obs.extract(py).unwrap();
//             let obs = obs.to_owned_array();
//             let obs = obs.mapv(|elem| elem as f32);
//             let obs = {
//                 if obs.shape().len() == S::shape().len() + 1 {
//                     // In this case obs has a dimension for n_procs
//                     obs
//                 }
//                 else if obs.shape().len() == S::shape().len() {
//                     // add dimension for n_procs
//                     obs.insert_axis(Axis(0))
//                 }
//                 else {
//                     panic!();
//                 }
//             };
//             Self {
//                 obs,
//                 phantom: PhantomData,
//             }
//         })
//     }
// }

// impl<S: Shape> From<PyObject> for TchPyGymEnvObs<S, f64>
// {
//     fn from(obs: PyObject) -> Self {
//         pyo3::Python::with_gil(|py| {
//             // let obs: &PyArrayDyn<f64> = obs.extract(py).unwrap();
//             let obs: &PyArrayDyn<f64> = obs.extract(py).unwrap();
//             let obs = obs.to_owned_array();
//             let obs = obs.mapv(|elem| elem as f32);
//             let obs = {
//                 if obs.shape().len() == S::shape().len() + 1 {
//                     // In this case obs has a dimension for n_procs
//                     obs
//                 }
//                 else if obs.shape().len() == S::shape().len() {
//                     // add dimension for n_procs
//                     obs.insert_axis(Axis(0))
//                 }
//                 else {
//                     panic!();
//                 }
//             };
//             Self {
//                 obs,
//                 phantom: PhantomData,
//             }
//         })
//     }
// }

// impl<S: Shape> From<PyObject> for TchPyGymEnvObs<S, u8>
// {
//     fn from(obs: PyObject) -> Self {
//         pyo3::Python::with_gil(|py| {
//             let obs: &PyArrayDyn<u8> = obs.extract(py).unwrap();
//             let obs = obs.to_owned_array();
//             let obs = obs.mapv(|elem| elem as f32);
//             let obs = {
//                 if obs.shape().len() == S::shape().len() + 1 {
//                     // In this case obs has a dimension for n_procs
//                     obs
//                 }
//                 else if obs.shape().len() == S::shape().len() {
//                     // add dimension for n_procs
//                     obs.insert_axis(Axis(0))
//                 }
//                 else {
//                     println!("{:?}", obs.shape());
//                     panic!();
//                 }
//             };
//             Self {
//                 obs,
//                 phantom: PhantomData,
//             }
//         })
//     }
// }

// impl <S, T> tch::kind::Element for TchPyGymEnvObs<S, T> where
//     S: Shape,
//     T: Element + Debug,
// {
//     const KIND: tch::Kind = tch::Kind::Float;
// }

// impl<S, T> Into<Tensor> for TchPyGymEnvObs<S, T> where
//     Self: Obs + Debug,
//     S: Shape,
//     T: Element + Debug,
// {
//     fn into(self) -> Tensor {
//         try_from(self.obs).unwrap()
//     }
// }

impl<S, T> From<TchPyGymEnvObs<S, T>> for Tensor where
    S: Shape,
    T: Element + Debug,
{
    fn from(v: TchPyGymEnvObs<S, T>) -> Tensor {
        try_from(v.obs).unwrap()
    }
}

pub struct TchPyGymEnvObsBuffer<S, T> where
    S: Shape,
    T: Element + Debug,
{
    obs: Tensor,
    phantom: PhantomData<(S, T)>,
}

impl<S, T> TchBuffer for TchPyGymEnvObsBuffer<S, T> where
    S: Shape,
    T: Element + Debug,
{
    type Item = TchPyGymEnvObs<S, T>;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        let capacity = capacity as _;
        let n_procs = n_procs as _;
        let shape = concat_slices(&[capacity, n_procs],
            S::shape().iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice());
        Self {
            obs: Tensor::zeros(&shape, tch::kind::FLOAT_CPU),
            phantom: PhantomData
        }
    }

    fn push(&mut self, index: i64, item: &Self::Item) {
        let obs = item.clone().into();
        self.obs.get(index).copy_(&obs);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        let batch = self.obs.index_select(0, &batch_indexes);
        batch.flatten(0, 1)
    }
}
