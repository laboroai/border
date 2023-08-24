use crate::GymActFilter;
use border_core::{
    record::{Record, RecordValue},
    Act,
};
use pyo3::{IntoPy, PyObject};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Serialize, Deserialize)]

/// Configuration of [`DiscreteActFilter`].
#[derive(Clone)]
pub struct DiscreteActFilterConfig {
    vectorized: bool,
}

impl Default for DiscreteActFilterConfig {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

/// Convert discrete action to PyObject.
///
/// Type `A` must be able to be converted into `Vec<i32>`.
#[derive(Clone, Debug)]
pub struct DiscreteActFilter<A> {
    phantom: PhantomData<A>,
}

impl<A> Default for DiscreteActFilter<A> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<A> GymActFilter<A> for DiscreteActFilter<A>
where
    A: Act + Into<Vec<i32>>,
{
    type Config = DiscreteActFilterConfig;

    fn build(_config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    fn filt(&mut self, act: A) -> (PyObject, Record) {
        let act = act.into();
        let record = Record::from_slice(&[(
            "act",
            RecordValue::Array1(act.iter().map(|v| *v as f32).collect::<Vec<_>>()),
        )]);

        // Commented out vectorized env support
        // let act = if self.vectorized {
        //     pyo3::Python::with_gil(|py| act.act.into_py(py))
        // } else {
        //     pyo3::Python::with_gil(|py| act.act[0].into_py(py))
        // };

        let act = pyo3::Python::with_gil(|py| act[0].into_py(py));

        (act, record)
    }
}
