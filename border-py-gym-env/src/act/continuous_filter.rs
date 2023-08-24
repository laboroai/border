use crate::util::arrayd_to_pyobj;
use crate::GymActFilter;
use border_core::{
    record::{Record, RecordValue},
    Act,
};
use ndarray::ArrayD;
use pyo3::PyObject;
use serde::{Deserialize, Serialize};
use std::{default::Default, fmt::Debug, marker::PhantomData};

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [`PyGymEnvContinuousActRawFilter`].
#[derive(Clone)]
pub struct ContinuousActFilterConfig {}

impl Default for ContinuousActFilterConfig {
    fn default() -> Self {
        Self {}
    }
}

/// Raw filter for continuous actions.
///
/// Type `A` must implements `Into<ArrayD<f32>>`
#[derive(Clone, Debug)]
pub struct ContinuousActFilter<A> {
    // `true` indicates that this filter is used in a vectorized environment.
    // pub vectorized: bool,
    phantom: PhantomData<A>,
}

impl<A> Default for ContinuousActFilter<A>
where
    A: Act + Into<ArrayD<f32>>,
{
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<A> GymActFilter<A> for ContinuousActFilter<A>
where
    A: Act + Into<ArrayD<f32>>,
{
    type Config = ContinuousActFilterConfig;

    fn build(_config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    /// Convert the given action to [PyObject].
    ///
    /// The first element of the shape of `act` is the batch dimension.
    fn filt(&mut self, act: A) -> (PyObject, Record) {
        let act: ArrayD<f32> = act.into();
        let record =
            Record::from_slice(&[("act", RecordValue::Array1(act.iter().cloned().collect()))]);
        let act = arrayd_to_pyobj(act);
        (act, record)
    }
}
