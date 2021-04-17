//! Items for interaction of PyGymEnvObs and tch agents.
use std::fmt::Debug;
use std::marker::PhantomData;
use log::trace;
use numpy::Element;
use tch::{Device, Tensor};

use crate::{
    agent::tch::{TchBuffer, util::concat_slices},
    env::py_gym_env::{Shape, obs::PyGymEnvObs, tch::util::try_from},
};

impl<S, T1, T2> From<PyGymEnvObs<S, T1, T2>> for Tensor where
    S: Shape,
    T1: Element + Debug,
    T2: 'static + Copy + tch::kind::Element
{
    fn from(v: PyGymEnvObs<S, T1, T2>) -> Tensor {
        try_from(v.obs).unwrap()
    }
}

/// Adds capability of constructing [Tensor] with a static method.
pub trait ZeroTensor {
    /// Constructs zero tensor.
    fn zeros(shape: &[i64]) -> Tensor;
}

impl ZeroTensor for u8 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(&shape, (tch::kind::Kind::Uint8, Device::Cpu))
    }
}

impl ZeroTensor for f32 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(&shape, tch::kind::FLOAT_CPU)
    }
}

/// Buffer of observations used in a replay buffer.
pub struct TchPyGymEnvObsBuffer<S, T1, T2> where
    S: Shape,
    T1: Element + Debug,
{
    obs: Tensor,
    phantom: PhantomData<(S, T1, T2)>,
}

impl<S, T1, T2> TchBuffer for TchPyGymEnvObsBuffer<S, T1, T2> where
    S: Shape,
    T1: Element + Debug,
    T2: 'static + Copy + tch::kind::Element + ZeroTensor,
{
    type Item = PyGymEnvObs<S, T1, T2>;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        let capacity = capacity as _;
        let n_procs = n_procs as _;
        let shape = concat_slices(&[capacity, n_procs],
            S::shape().iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice());
        Self {
            obs: T2::zeros(&shape),
            phantom: PhantomData
        }
    }

    fn push(&mut self, index: i64, item: &Self::Item) {
        trace!("TchPyGymEnvObsBuffer::push()");

        let obs: Tensor = item.clone().into();
        debug_assert_eq!(obs.size().as_slice(), &self.obs.size()[1..]);
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
