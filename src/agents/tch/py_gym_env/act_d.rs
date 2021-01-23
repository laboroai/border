use std::marker::PhantomData;
use std::fmt::Debug;
use std::convert::TryFrom;
use log::trace;
use pyo3::{PyObject, IntoPy};
use tch::Tensor;
use crate::core::Act;
use crate::agents::tch::TchBuffer;

/// Filtering action before applied to the environment.
pub trait TchPyGymDiscreteActFilter: Clone + Debug {
    fn filt(act: Vec<i32>) -> Vec<i32> {
        act
    }
}

#[derive(Clone, Debug)]
pub struct TchPyGymEnvDiscreteActRawFilter {}

impl TchPyGymDiscreteActFilter for TchPyGymEnvDiscreteActRawFilter {}

/// Represents action.
#[derive(Clone, Debug)]
pub struct TchPyGymEnvDiscreteAct<F: TchPyGymDiscreteActFilter> {
    act: Vec<i32>,
    phantom: PhantomData<F>
}

impl<F: TchPyGymDiscreteActFilter> TchPyGymEnvDiscreteAct<F> {
    pub fn new(act: Vec<i32>) -> Self {
        Self {
            act,
            phantom: PhantomData
        }
    }
}

impl<F: TchPyGymDiscreteActFilter> Act for TchPyGymEnvDiscreteAct<F> {}

impl<F: TchPyGymDiscreteActFilter> Into<PyObject> for TchPyGymEnvDiscreteAct<F> {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            F::filt(self.act).into_py(py)
        })
    }
}

impl<F: TchPyGymDiscreteActFilter> From<Tensor> for TchPyGymEnvDiscreteAct<F> {
    /// Assumes `t` is a scalar or 1-dimensional vector,
    /// containing discrete actions. The number of elements is
    /// equal to the number of environments in the vectorized environment.
    fn from(t: Tensor) -> Self {
        trace!("Tensor from TchPyGymEnvDiscreteAct: {:?}", t);
        Self {
            act: t.into(),
            phantom: PhantomData
        }
    }
}

pub struct TchPyGymEnvDiscreteActBuffer<F: TchPyGymDiscreteActFilter> {
    act: Tensor,
    n_procs: i64,
    phantom: PhantomData<F>,
}

impl<F: TchPyGymDiscreteActFilter> TchBuffer for TchPyGymEnvDiscreteActBuffer<F> {
    type Item = TchPyGymEnvDiscreteAct<F>;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        Self {
            act: Tensor::zeros(&[capacity as _, n_procs as _], tch::kind::INT64_CPU),
            n_procs: n_procs as _,
            phantom: PhantomData
        }
    }

    fn push(&mut self, index: i64, item: &TchPyGymEnvDiscreteAct<F>) {
        let act = Tensor::try_from(item.act.clone()).unwrap();
        trace!("TchPyGymDiscreteActBuffer.push(): {:?}", act);
        self.act.get(index).copy_(&act);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        let batch = self.act.index_select(0, &batch_indexes);
        let batch = batch.flatten(0, 1).unsqueeze(-1);
        debug_assert!(batch.size().as_slice() == [batch_indexes.size()[0], self.n_procs]);
        batch
    }
}
