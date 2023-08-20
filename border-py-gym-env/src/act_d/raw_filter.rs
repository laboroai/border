use super::GymDiscreteAct;
use crate::GymActFilter;
use border_core::{
    record::{Record, RecordValue},
    Act,
};
use pyo3::{IntoPy, PyObject};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [`PyGymEnvDiscreteActRawFilter`].
#[derive(Clone)]
pub struct PyGymEnvDiscreteActRawFilterConfig {
    vectorized: bool,
}

impl Default for PyGymEnvDiscreteActRawFilterConfig {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

/// Raw filter for discrete actions.
///
/// No processing is applied to actions.
#[derive(Clone, Debug)]
pub struct GymDiscreteActRawFilter<T> {
    /// `true` for filters on vectorized environments.
    pub vectorized: bool,
    phantom: PhantomData<T>,
}

impl<T> GymDiscreteActRawFilter<T>
where
    T: Act + Into<GymDiscreteAct>,
{
    /// Returns `true` for filters working with vectorized environments.
    pub fn vectorized() -> Self {
        Self {
            vectorized: true,
            phantom: PhantomData,
        }
    }
}

impl<T> Default for GymDiscreteActRawFilter<T> {
    fn default() -> Self {
        Self {
            vectorized: false,
            phantom: PhantomData,
        }
    }
}

// TODO: support vecenv
impl<T> GymActFilter<T> for GymDiscreteActRawFilter<T>
where
    T: Act + Into<GymDiscreteAct>,
{
    type Config = PyGymEnvDiscreteActRawFilterConfig;

    fn build(config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            vectorized: config.vectorized,
            phantom: PhantomData,
        })
    }

    fn filt(&mut self, act: T) -> (PyObject, Record) {
        let act = act.into();
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
