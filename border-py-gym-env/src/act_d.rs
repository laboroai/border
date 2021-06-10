//! Discrete action for [`super::PyGymEnv`] and [`super::PyVecGymEnv`].
use crate::PyGymEnvActFilter;
use border_core::{
    record::{Record, RecordValue},
    Act,
};
use pyo3::{IntoPy, PyObject};
use std::default::Default;
use std::fmt::Debug;

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteAct {
    pub act: Vec<i32>,
}

impl PyGymEnvDiscreteAct {
    /// Constructs a discrete action.
    pub fn new(act: Vec<i32>) -> Self {
        Self { act }
    }
}

impl Act for PyGymEnvDiscreteAct {}

/// Raw filter for discrete actions.
///
/// No processing is applied to actions.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteActRawFilter {
    /// `true` for filters on vectorized environments.
    pub vectorized: bool,
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

// TODO: support vecenv
impl PyGymEnvActFilter<PyGymEnvDiscreteAct> for PyGymEnvDiscreteActRawFilter {
    fn filt(&mut self, act: PyGymEnvDiscreteAct) -> (PyObject, Record) {
        let record = Record::from_slice(&[(
            "act",
            RecordValue::Array1(act.act.iter().map(|v| *v as f32).collect::<Vec<_>>()),
        )]);

        let act = if self.vectorized {
            pyo3::Python::with_gil(|py| act.act.into_py(py))
        } else {
            pyo3::Python::with_gil(|py| act.act[0].into_py(py))
        };
        (act, record)
    }
}

/// Defines newtypes of [PyGymEnvDiscreteAct] and [PyGymEnvDiscreteActRawFilter].
///
/// TODO: add example.
#[macro_export]
macro_rules! newtype_act_d {
    ($struct_:ident) => {
        #[derive(Clone, Debug)]
        struct $struct_(border_py_gym_env::PyGymEnvDiscreteAct);

        impl $struct_ {
            fn new(act: Vec<i32>) -> Self {
                $struct_(border_py_gym_env::PyGymEnvDiscreteAct::new(act))
            }
        }

        impl border_core::Act for $struct_ {}
    };
    ($struct_:ident, $struct2_:ident) => {
        newtype_act_d!($struct_);

        struct $struct2_(border_py_gym_env::PyGymEnvDiscreteActRawFilter);

        impl border_py_gym_env::PyGymEnvActFilter<$struct_> for $struct2_ {
            fn filt(&mut self, act: $struct_) -> (pyo3::PyObject, border_core::record::Record) {
                self.0.filt(act.0)
            }
        }

        impl std::default::Default for $struct2_ {
            fn default() -> Self {
                Self(border_py_gym_env::PyGymEnvDiscreteActRawFilter::default())
            }
        }
    };
}
