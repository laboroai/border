//! Configuration of [SimpleReplayBuffer](super::SimpleReplayBuffer).
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};
use super::{WeightNormalizer, WeightNormalizer::{All, Batch}};

/// Configuration of [PerState](super::base::PerState).
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct PerConfig {
    pub(super) alpha: f32,
    /// Initial value of $\beta$.
    pub(super) beta_0: f32,
    /// Final value of $\beta$.
    pub(super) beta_final: f32,
    /// Optimization step when beta reaches its final value.
    pub(super) n_opts_final: usize,
    /// How to normalize the weights.
    pub(super) normalize: WeightNormalizer,
}

impl Default for PerConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6,
            beta_0: 0.4,
            beta_final: 1.0,
            n_opts_final: 500_000,
            normalize: All,
        }
    }
}

impl PerConfig {
    /// Sets alpha, the exponent of sampling probability.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets beta_0, the initial value of the exponent of importance weights.
    pub fn beta_0(mut self, beta_0: f32) -> Self {
        self.beta_0 = beta_0;
        self
    }

    /// Sets beta_final, the final value of the exponent of importance weights.
    pub fn beta_final(mut self, beta_final: f32) -> Self {
        self.beta_final = beta_final;
        self
    }

    /// Sets the optimization step when beta reaches the final value.
    pub fn n_opts_final(mut self, n_opts_final: usize) -> Self {
        self.n_opts_final = n_opts_final;
        self
    }

    /// Sets how to normalize the importance weight.
    pub fn normalize(mut self, normalize: WeightNormalizer) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Configuration of [SimpleReplayBuffer](super::SimpleReplayBuffer).
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct SimpleReplayBufferConfig {
    pub(super) capacity: usize,
    pub(super) seed: u64,
    pub(super) per_config: Option<PerConfig>,
}

impl Default for SimpleReplayBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            seed: 42,
            per_config: None,
        }
    }
}

impl SimpleReplayBufferConfig {
    /// Sets the capacity of the replay buffer.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets configuration of PER.
    pub fn per_config(mut self, per_config: Option<PerConfig>) -> Self {
        self.per_config = per_config;
        self
    }

    /// Constructs [SimpleReplayBufferConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [SimpleReplayBufferConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
