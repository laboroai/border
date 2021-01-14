use std::convert::TryFrom;
use log::trace;
use pyo3::{PyObject, IntoPy};
use tch::Tensor;
use crate::core::Act;
use crate::agents::tch::TchBuffer;

#[derive(Clone, Debug)]
pub struct TchPyGymEnvContinuousAct (Vec<f32>);

impl TchPyGymEnvContinuousAct {
    pub fn new(v: Vec<f32>) -> Self {
        Self(v)
    }
}

impl Act for TchPyGymEnvContinuousAct {}

impl Into<PyObject> for TchPyGymEnvContinuousAct {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            self.0.into_py(py)
        })
    }
}

impl From<Tensor> for TchPyGymEnvContinuousAct {
    /// The first dimension is the number of environments.
    fn from(t: Tensor) -> Self {
        trace!("Tensor from TchPyGymEnvContinuousAct: {:?}", t);
        Self(t.into())
    }
}

pub struct TchPyGymEnvContinuousActBuffer {
    act: Tensor
}

impl TchBuffer for TchPyGymEnvContinuousActBuffer {
    type Item = TchPyGymEnvContinuousAct;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        Self {
            act: Tensor::zeros(&[capacity as _, n_procs as _], tch::kind::INT64_CPU),
        }
    }

    fn push(&mut self, index: i64, item: &TchPyGymEnvContinuousAct) {
        let act = Tensor::try_from(item.0.clone()).unwrap();
        trace!("TchPyGymDiscreteActBuffer.push(): {:?}", act);
        self.act.get(index).copy_(&act);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        self.act.index_select(0, &batch_indexes)
    }
}
