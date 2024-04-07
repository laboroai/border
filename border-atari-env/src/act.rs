//! Action for [BorderAtariEnv](crate::BorderAtariEnv)
use anyhow::Result;
use border_core::{record::Record, Act};
use serde::{Deserialize, Serialize};
use std::{default::Default, marker::PhantomData};

#[derive(Debug, Clone)]
/// Action for [BorderAtariEnv](crate::BorderAtariEnv)
pub struct BorderAtariAct {
    pub act: u8,
}

impl BorderAtariAct {
    pub fn new(act: u8) -> Self {
        Self { act }
    }
}

impl Act for BorderAtariAct {
    fn len(&self) -> usize {
        1
    }
}

impl From<u8> for BorderAtariAct {
    fn from(act: u8) -> Self {
        Self { act }
    }
}

/// Converts `A` to [`BorderAtariAct`].
pub trait BorderAtariActFilter<A: Act> {
    /// Configuration of the filter.
    type Config: Clone + Default;

    /// Constructs the filter given a configuration.
    fn build(config: &Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Converts `A` into an action of [BorderAtariAct].
    fn filt(&mut self, act: A) -> (BorderAtariAct, Record);

    /// Resets the filter. Does nothing in the default implementation.
    fn reset(&mut self, _is_done: &Option<&Vec<i8>>) {}
}

#[derive(Debug, Deserialize, Serialize)]
/// Configuration of [`BorderAtariActRawFilter`].
#[derive(Clone)]
pub struct BorderAtariActRawFilterConfig;

impl Default for BorderAtariActRawFilterConfig {
    fn default() -> Self {
        Self
    }
}

/// A filter without any processing.
pub struct BorderAtariActRawFilter<A> {
    phantom: PhantomData<A>,
}

impl<A> BorderAtariActFilter<A> for BorderAtariActRawFilter<A>
where
    A: Act + Into<BorderAtariAct>,
{
    type Config = BorderAtariActRawFilterConfig;

    fn build(_config: &Self::Config) -> Result<Self> {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    fn filt(&mut self, act: A) -> (BorderAtariAct, Record) {
        (act.into(), Record::empty())
    }
}
