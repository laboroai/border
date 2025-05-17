//! Configuration for the replay buffer implementation.
//!
//! This module provides configuration structures for the replay buffer, including:
//! - Basic buffer configuration (capacity, seed)
//! - Prioritized Experience Replay (PER) configuration
//! - Serialization and deserialization support

use super::{WeightNormalizer, WeightNormalizer::All};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration for Prioritized Experience Replay (PER).
///
/// This structure defines the parameters for prioritized sampling in the replay buffer.
/// It controls how transitions are sampled based on their importance and how
/// importance weights are calculated and normalized.
///
/// # Fields
///
/// * `alpha` - Controls the degree of prioritization (0 = uniform sampling)
/// * `beta_0` - Initial value for importance sampling weights
/// * `beta_final` - Final value for importance sampling weights
/// * `n_opts_final` - Number of optimization steps to reach `beta_final`
/// * `normalize` - Method for normalizing importance weights
///
/// # Examples
///
/// ```rust
/// use border_core::generic_replay_buffer::{PerConfig, WeightNormalizer};
///
/// let config = PerConfig::default()
///     .alpha(0.6)
///     .beta_0(0.4)
///     .beta_final(1.0)
///     .n_opts_final(500_000)
///     .normalize(WeightNormalizer::All);
/// ```
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct PerConfig {
    /// Exponent for prioritization. Higher values increase the bias towards
    /// high-priority transitions. A value of 0 results in uniform sampling.
    pub alpha: f32,

    /// Initial value of the importance sampling exponent. Lower values reduce
    /// the impact of importance sampling weights.
    pub beta_0: f32,

    /// Final value of the importance sampling exponent. Typically set to 1.0
    /// to fully compensate for the non-uniform sampling.
    pub beta_final: f32,

    /// Number of optimization steps after which `beta` reaches its final value.
    /// This allows for a gradual increase in the impact of importance sampling.
    pub n_opts_final: usize,

    /// Method for normalizing importance sampling weights. Controls how the
    /// weights are scaled to prevent numerical instability.
    pub normalize: WeightNormalizer,
}

impl Default for PerConfig {
    /// Creates a default PER configuration with commonly used values:
    /// - `alpha = 0.6` (moderate prioritization)
    /// - `beta_0 = 0.4` (initial importance sampling)
    /// - `beta_final = 1.0` (full compensation)
    /// - `n_opts_final = 500_000` (gradual increase)
    /// - `normalize = All` (normalize all weights)
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
    /// Sets the prioritization exponent `alpha`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The new value for the prioritization exponent
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the initial importance sampling exponent `beta_0`.
    ///
    /// # Arguments
    ///
    /// * `beta_0` - The new initial value for the importance sampling exponent
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn beta_0(mut self, beta_0: f32) -> Self {
        self.beta_0 = beta_0;
        self
    }

    /// Sets the final importance sampling exponent `beta_final`.
    ///
    /// # Arguments
    ///
    /// * `beta_final` - The new final value for the importance sampling exponent
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn beta_final(mut self, beta_final: f32) -> Self {
        self.beta_final = beta_final;
        self
    }

    /// Sets the number of optimization steps to reach the final beta value.
    ///
    /// # Arguments
    ///
    /// * `n_opts_final` - The new number of optimization steps
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn n_opts_final(mut self, n_opts_final: usize) -> Self {
        self.n_opts_final = n_opts_final;
        self
    }

    /// Sets the method for normalizing importance weights.
    ///
    /// # Arguments
    ///
    /// * `normalize` - The new normalization method
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn normalize(mut self, normalize: WeightNormalizer) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Configuration for the replay buffer.
///
/// This structure defines the basic parameters for the replay buffer,
/// including its capacity, random seed, and optional PER configuration.
///
/// # Fields
///
/// * `capacity` - Maximum number of transitions to store
/// * `seed` - Random seed for sampling
/// * `per_config` - Optional configuration for prioritized experience replay
///
/// # Examples
///
/// ```rust
/// use border_core::generic_replay_buffer::{SimpleReplayBufferConfig, PerConfig};
///
/// // Basic configuration
/// let config = SimpleReplayBufferConfig::default()
///     .capacity(10000)
///     .seed(42);
///
/// // Configuration with PER
/// let config_with_per = SimpleReplayBufferConfig::default()
///     .capacity(10000)
///     .seed(42)
///     .per_config(Some(PerConfig::default()));
/// ```
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct SimpleReplayBufferConfig {
    /// Maximum number of transitions that can be stored in the buffer.
    /// When the buffer is full, new transitions replace the oldest ones.
    pub capacity: usize,

    /// Random seed used for sampling transitions. This ensures reproducibility
    /// of the sampling process when the same seed is used.
    pub seed: u64,

    /// Optional configuration for prioritized experience replay. If `None`,
    /// transitions are sampled uniformly at random.
    pub per_config: Option<PerConfig>,
}

impl Default for SimpleReplayBufferConfig {
    /// Creates a default replay buffer configuration with commonly used values:
    /// - `capacity = 10000` (moderate buffer size)
    /// - `seed = 42` (fixed random seed)
    /// - `per_config = None` (uniform sampling)
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
    ///
    /// # Arguments
    ///
    /// * `capacity` - The new capacity for the buffer
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets the random seed for sampling.
    ///
    /// # Arguments
    ///
    /// * `seed` - The new random seed
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the configuration for prioritized experience replay.
    ///
    /// # Arguments
    ///
    /// * `per_config` - The new PER configuration
    ///
    /// # Returns
    ///
    /// The modified configuration
    pub fn per_config(mut self, per_config: Option<PerConfig>) -> Self {
        self.per_config = per_config;
        self
    }

    /// Loads the configuration from a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// The loaded configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves the configuration to a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the configuration should be saved
    ///
    /// # Returns
    ///
    /// `Ok(())` if the configuration was saved successfully
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
