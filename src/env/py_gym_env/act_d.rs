//! Discrete action for [`super::PyGymEnv`] and [`super::PyVecGymEnv`].
use std::fmt::Debug;
use std::default::Default;
use pyo3::{PyObject, IntoPy};

use crate::{
    core::{Act, record::{Record, RecordValue}},
    env::py_gym_env::PyGymEnvActFilter
};

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

/// Raw filter for discrete actions.
///
/// No processing is applied to actions.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteActRawFilter {
    /// `true` for filters on vectorized environments.
    pub vectorized: bool
}

impl PyGymEnvDiscreteActRawFilter {
    /// Returns `true` for filters working with vectorized environments.
    pub fn vectorized() -> Self {
        Self { vectorized: true }
    }
}

impl Default for PyGymEnvDiscreteActRawFilter {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

impl PyGymEnvDiscreteActRawFilter {}

// TODO: support vecenv
impl PyGymEnvActFilter<PyGymEnvDiscreteAct> for PyGymEnvDiscreteActRawFilter {
    fn filt(&mut self, act: PyGymEnvDiscreteAct) -> (PyObject, Record) {
        let record = Record::from_slice(&[
            ("act", RecordValue::Array1(
                act.act.iter().map(|v| *v as f32).collect::<Vec<_>>()
            ))
        ]);

        let act = if self.vectorized {
            pyo3::Python::with_gil(|py| {
                act.act.into_py(py)
            })
        }
        else {
            pyo3::Python::with_gil(|py| {
                act.act[0].into_py(py)
            })
        };
        (act, record)
    }
}
