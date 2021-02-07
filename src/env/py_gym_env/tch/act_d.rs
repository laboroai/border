// use std::marker::PhantomData;
use std::convert::TryFrom;
use log::trace;
use tch::Tensor;
use crate::agents::tch::TchBuffer;
use crate::py_gym_env::act_d::PyGymEnvDiscreteAct; //, PyGymDiscreteActFilter};

impl From<Tensor> for PyGymEnvDiscreteAct {
    /// Assumes `t` is a scalar or 1-dimensional vector,
    /// containing discrete actions. The number of elements is
    /// equal to the number of environments in the vectorized environment.
    fn from(t: Tensor) -> Self {
        trace!("Tensor from TchPyGymEnvDiscreteAct: {:?}", t);
        Self {
            act: t.into(),
        }
    }
}

pub struct TchPyGymEnvDiscreteActBuffer {
    act: Tensor,
    n_procs: i64,
}

impl TchBuffer for TchPyGymEnvDiscreteActBuffer {
    type Item = PyGymEnvDiscreteAct;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        Self {
            act: Tensor::zeros(&[capacity as _, n_procs as _], tch::kind::INT64_CPU),
            n_procs: n_procs as _,
        }
    }

    fn push(&mut self, index: i64, item: &PyGymEnvDiscreteAct) {
        let act = Tensor::try_from(item.act.clone()).unwrap();
        trace!("TchPyGymDiscreteActBuffer::push(): {:?}", act);
        self.act.get(index).copy_(&act);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        let batch = self.act.index_select(0, &batch_indexes);
        let batch = batch.flatten(0, 1).unsqueeze(-1);
        debug_assert_eq!(batch.size().as_slice(), [batch_indexes.size()[0] * self.n_procs, 1]);
        batch
    }
}
