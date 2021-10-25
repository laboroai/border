//! Observation of [BorderAtariEnv](super::BorderAtariEnv).
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
use border_core::Obs;
#[cfg(feature = "tch")]
use {tch::Tensor, std::convert::TryFrom};

/// Observation of [BorderAtariEnv](super::BorderAtariEnv).
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

    fn merge(self, _obs_reset: Self, _is_done: &[i8]) -> Self {
        unimplemented!();
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
        Tensor::try_from(tmp).unwrap().reshape(&[1, 4, 84, 84])
    }
}
