use std::fmt::Debug;
use std::marker::PhantomData;
use log::trace;
use numpy::Element;
use tch::Tensor;
use crate::agents::tch::{Shape, TchBuffer, util::try_from, util::concat_slices};
use crate::py_gym_env::PyGymEnvObs;

impl<S, T> From<PyGymEnvObs<S, T>> for Tensor where
    S: Shape,
    T: Element + Debug,
{
    fn from(v: PyGymEnvObs<S, T>) -> Tensor {
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
    type Item = PyGymEnvObs<S, T>;
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
