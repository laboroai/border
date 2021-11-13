//! Configuration of [SimpleReplayBuffer](super::SimpleReplayBuffer).
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration of [PerState](super::base::PerState).
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct PerConfig {
    pub(super) alpha: f32,
    /// Initial value of $\beta$.
    pub(super) beta_0: f32,
    /// Final value of $\beta$.
    pub(super) beta_final: f32,
    /// Optimization steps when beta reaches its final value.
    pub(super) n_opts_final: usize,
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
