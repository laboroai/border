use crate::core::{Act};
use pyo3::{IntoPy, PyObject};

pub trait PyGymEnvAct: Act + Into<PyObject> {}

#[derive(Debug, Clone)]
pub struct PyGymDiscreteAct (pub(in crate::py_gym_env) u32);

impl PyGymDiscreteAct {
    pub fn new(v: u32) -> Self {
        PyGymDiscreteAct { 0: v }
    }
}

impl Act for PyGymDiscreteAct {}

impl PyGymEnvAct for PyGymDiscreteAct {}

impl Into<PyObject> for PyGymDiscreteAct {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            self.0.into_py(py)
        })
    }
}
