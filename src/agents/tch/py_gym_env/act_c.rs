// use std::convert::TryFrom;
use log::trace;
use pyo3::{PyObject, IntoPy};
use tch::Tensor;
use crate::core::Act;
// use crate::agents::tch::TchBuffer;

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
