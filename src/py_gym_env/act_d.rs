use std::fmt::Debug;
use std::default::Default;
use pyo3::{PyObject, IntoPy};
use crate::core::Act;
use crate::py_gym_env::PyGymEnvActFilter;

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteAct {
    pub(crate) act: Vec<i32>,
}

impl PyGymEnvDiscreteAct {
    pub fn new(act: Vec<i32>) -> Self {
        Self {
            act,
        }
    }
}

impl Act for PyGymEnvDiscreteAct {}

// impl<F: PyGymDiscreteActFilter> Into<PyObject> for PyGymEnvDiscreteAct<F> {
//     fn into(self) -> PyObject {
//         pyo3::Python::with_gil(|py| {
//             F::filt(self.act).into_py(py)
//         })
//     }
// }

/// Filter action before applied to the environment.
pub trait PyGymDiscreteActFilter: Clone + Debug {
    fn filt(act: Vec<i32>) -> Vec<i32> {
        act
    }
}

#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteActRawFilter {
    pub vectorized: bool
}

impl Default for PyGymEnvDiscreteActRawFilter {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

impl PyGymEnvDiscreteActRawFilter {}

// TODO: support vecenv
impl PyGymEnvActFilter<PyGymEnvDiscreteAct> for PyGymEnvDiscreteActRawFilter {
    fn filt(&mut self, act: PyGymEnvDiscreteAct) -> PyObject {
        if self.vectorized {
            pyo3::Python::with_gil(|py| {
                act.act.into_py(py)
            })
        }
        else {
            pyo3::Python::with_gil(|py| {
                act.act[0].into_py(py)
            })
        }
    }
}
