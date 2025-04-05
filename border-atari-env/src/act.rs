//! Action for [BorderAtariEnv](crate::BorderAtariEnv)
use anyhow::Result;
use border_core::{record::Record, Act};
use serde::{Deserialize, Serialize};
use std::{default::Default, marker::PhantomData};

#[derive(Debug, Clone)]
/// Action for [`BorderAtariEnv`](crate::BorderAtariEnv).
///
/// This action is a discrete action and denotes pushing a button.
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

#[cfg(feature = "candle")]
pub mod candle {
    use super::*;
    use border_candle_agent::TensorBatch;
    use candle_core::{Device::Cpu, Tensor};

    impl From<BorderAtariAct> for Tensor {
        fn from(act: BorderAtariAct) -> Tensor {
            Tensor::from_vec(vec![act.act as u8], &[1, 1], &Cpu).unwrap()
        }
    }

    impl From<BorderAtariAct> for TensorBatch {
        fn from(act: BorderAtariAct) -> Self {
            let tensor = act.into();
            TensorBatch::from_tensor(tensor)
        }
    }

    impl From<Tensor> for BorderAtariAct {
        /// `t` must have single item.
        fn from(t: Tensor) -> Self {
            (t.to_vec1::<i64>().unwrap()[0] as u8).into()
        }
    }
}

#[cfg(feature = "tch")]
pub mod tch_ {
    use super::*;
    use border_tch_agent::TensorBatch;
    use tch::Tensor;
    use std::convert::TryInto;

    impl From<BorderAtariAct> for Tensor {
        fn from(act: BorderAtariAct) -> Tensor {
            Tensor::from_slice(&[act.act as i64])
        }
    }

    impl From<BorderAtariAct> for TensorBatch {
        fn from(act: BorderAtariAct) -> Self {
            let tensor = act.into();
            TensorBatch::from_tensor(tensor)
        }
    }

    impl From<Tensor> for BorderAtariAct {
        /// `t` must have single item.
        fn from(t: Tensor) -> Self {
            (TryInto::<i64>::try_into(t).unwrap() as u8).into()
        }
    }
}

/// Converts action of type `A` to [`BorderAtariAct`].
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

/// A filter that performs no processing.
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
