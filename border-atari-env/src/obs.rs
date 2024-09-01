//! Observation of [`BorderAtariEnv`](super::BorderAtariEnv).
//!
//! It applies the following preprocessing
//! (explanations are adapted from [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.AtariWrapper)
//! API document):
//! * (WIP: NoopReset: obtain initial state by taking random number of no-ops on reset.)
//! * Four frames skipping
//! * Max pooling: most recent two observations
//! * Resize to 84 x 84
//! * Grayscale
//! * Clip reward to {-1, 0, 1} in training
//! * Stacking four frames
//! It does not apply pixel scaling from 255 to 1.0 for saving memory of the replay buffer.
//! Instead, the scaling is applied in CNN model.
use anyhow::Result;
use border_core::{record::Record, Obs};
#[cfg(feature = "candle-core")]
use candle_core::{Device::Cpu, Tensor};
use serde::{Deserialize, Serialize};
use std::{default::Default, marker::PhantomData};
#[cfg(feature = "tch")]
use {std::convert::TryFrom, tch::Tensor};

/// Observation of [`BorderAtariEnv`](super::BorderAtariEnv).
#[derive(Debug, Clone)]
pub struct BorderAtariObs {
    /// Four frames of 84 * 84 pixels.
    pub frames: Vec<u8>,
}

impl From<Vec<u8>> for BorderAtariObs {
    fn from(frames: Vec<u8>) -> Self {
        Self { frames }
    }
}

impl Obs for BorderAtariObs {
    fn dummy(_n: usize) -> Self {
        Self {
            frames: vec![0; 4 * 84 * 84],
        }
    }

    fn len(&self) -> usize {
        1
    }
}

#[cfg(feature = "tch")]
impl From<BorderAtariObs> for Tensor {
    fn from(obs: BorderAtariObs) -> Tensor {
        let tmp = &obs.frames;
        // Assumes the batch size is 1, implying non-vectorized environment
        Tensor::try_from(tmp).unwrap().reshape(&[1, 4, 1, 84, 84])
    }
}

#[cfg(feature = "candle-core")]
impl From<BorderAtariObs> for Tensor {
    fn from(obs: BorderAtariObs) -> Tensor {
        let tmp = obs.frames;
        // Assumes the batch size is 1, implying non-vectorized environment
        Tensor::from_vec(tmp, &[1 * 4 * 1 * 84 * 84], &Cpu)
            .unwrap()
            .reshape(&[1, 4, 1, 84, 84])
            .unwrap()
    }
}

/// Converts [`BorderAtariObs`] to observation of type `O` with an arbitrary processing.
pub trait BorderAtariObsFilter<O: Obs> {
    /// Configuration of the filter.
    type Config: Clone + Default;

    /// Constructs the filter given a configuration.
    fn build(config: &Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Converts the original observation into `O`.
    fn filt(&mut self, obs: BorderAtariObs) -> (O, Record);

    /// Resets the filter.
    fn reset(&mut self, obs: BorderAtariObs) -> O {
        let (obs, _) = self.filt(obs);
        obs
    }
}

#[derive(Serialize, Deserialize, Debug)]
/// Configuration of [`BorderAtariObsRawFilter`].
#[derive(Clone)]
pub struct BorderAtariObsRawFilterConfig;

impl Default for BorderAtariObsRawFilterConfig {
    fn default() -> Self {
        Self
    }
}

/// A filter that performs no processing.
pub struct BorderAtariObsRawFilter<O> {
    phantom: PhantomData<O>,
}

impl<O> BorderAtariObsFilter<O> for BorderAtariObsRawFilter<O>
where
    O: Obs + From<BorderAtariObs>,
{
    type Config = BorderAtariObsRawFilterConfig;

    fn build(_config: &Self::Config) -> Result<Self> {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    fn filt(&mut self, obs: BorderAtariObs) -> (O, Record) {
        (obs.into(), Record::empty())
    }
}
