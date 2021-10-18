use super::{to_pyobj, PyGymEnvContinuousAct};
use crate::PyGymEnvActFilter;
use border_core::{
    record::{Record, RecordValue},
    Act, Shape,
};
use pyo3::PyObject;
use serde::{Deserialize, Serialize};
use std::{default::Default, fmt::Debug, marker::PhantomData};

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [PyGymEnvContinuousActRawFilter].
pub struct PyGymEnvContinuousActRawFilterConfig {
    vectorized: bool,
}

impl Default for PyGymEnvContinuousActRawFilterConfig {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

/// Raw filter for continuous actions.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousActRawFilter<S, T> {
    /// `true` indicates that this filter is used in a vectorized environment.
    pub vectorized: bool,
    phantom: PhantomData<(S, T)>,
}

impl<S, T> Default for PyGymEnvContinuousActRawFilter<S, T>
where
    T: Act + Into<PyGymEnvContinuousAct<S>>,
    S: Shape,
{
    fn default() -> Self {
        Self {
            vectorized: false,
            phantom: PhantomData,
        }
    }
}

impl<S, T> PyGymEnvActFilter<T> for PyGymEnvContinuousActRawFilter<S, T>
where
    T: Act + Into<PyGymEnvContinuousAct<S>>,
    S: Shape,
{
    type Config = PyGymEnvContinuousActRawFilterConfig;

    fn build(config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            vectorized: config.vectorized,
            phantom: PhantomData,
        })
    }

    /// Convert [PyGymEnvContinuousAct] to [PyObject].
    /// No processing will be applied to the action.
    ///
    /// The first element of the shape of `act.act` is batch dimension and
    /// `act.act.size()[1..]` is equal to S::shape().
    ///
    /// TODO: explain action representation for the vectorized environment.
    fn filt(&mut self, act: T) -> (PyObject, Record) {
        let act = act.into();
        let act = act.act;
        let record =
            Record::from_slice(&[("act", RecordValue::Array1(act.iter().cloned().collect()))]);
        let act = to_pyobj::<S>(act);
        (act, record)
    }
}
