//! Configuration for the training process.
//!
//! This module provides configuration options for controlling the training process
//! of reinforcement learning agents. It allows fine-tuning of various aspects
//! of the training loop, including optimization intervals, evaluation frequency,
//! and model saving.
//!
//! # Configuration Options
//!
//! The configuration allows control over:
//!
//! * Training duration and optimization steps
//! * Evaluation frequency and model selection
//! * Performance monitoring and metrics recording
//! * Model checkpointing and warmup periods
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration parameters for the training process.
///
/// This struct defines various intervals and thresholds that control the
/// behavior of the training loop. Each parameter can be set using the
/// builder pattern methods.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct TrainerConfig {
    /// Maximum number of optimization steps to perform.
    /// Training stops when this number is reached.
    pub max_opts: usize,

    /// Number of environment steps between optimization updates.
    /// For example, if set to 1, optimization occurs after every environment step.
    pub opt_interval: usize,

    /// Number of optimization steps between performance evaluations.
    /// During evaluation, the agent's performance is measured and the best model is saved.
    pub eval_interval: usize,

    /// Number of optimization steps between flushing recorded metrics to storage.
    /// This controls how frequently training metrics are persisted.
    pub flush_record_interval: usize,

    /// Number of optimization steps between recording computational performance metrics.
    /// This includes metrics like optimization steps per second.
    pub record_compute_cost_interval: usize,

    /// Number of optimization steps between recording agent-specific information.
    /// This can include internal agent metrics or state information.
    pub record_agent_info_interval: usize,

    /// Initial number of environment steps before optimization begins.
    /// During this period, the replay buffer is filled with initial experiences.
    pub warmup_period: usize,

    /// Number of optimization steps between saving model checkpoints.
    /// These checkpoints can be used for resuming training or analysis.
    pub save_interval: usize,
}

impl Default for TrainerConfig {
    /// Creates a default configuration with conservative values.
    ///
    /// Default values are set to:
    /// * `max_opts`: 0
    /// * `opt_interval`: 1 (optimize every step)
    /// * `eval_interval`: 0 (no evaluation)
    /// * `flush_record_interval`: usize::MAX (never flush)
    /// * `record_compute_cost_interval`: usize::MAX (never record)
    /// * `record_agent_info_interval`: usize::MAX (never record)
    /// * `warmup_period`: 0 (no warmup)
    /// * `save_interval`: usize::MAX (never save)
    fn default() -> Self {
        Self {
            max_opts: 0,
            eval_interval: 0,
            opt_interval: 1,
            flush_record_interval: usize::MAX,
            record_compute_cost_interval: usize::MAX,
            record_agent_info_interval: usize::MAX,
            warmup_period: 0,
            save_interval: usize::MAX,
        }
    }
}

impl TrainerConfig {
    /// Sets the maximum number of optimization steps.
    ///
    /// # Arguments
    ///
    /// * `v` - Maximum number of optimization steps
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn max_opts(mut self, v: usize) -> Self {
        self.max_opts = v;
        self
    }

    /// Sets the interval between performance evaluations.
    ///
    /// # Arguments
    ///
    /// * `v` - Number of optimization steps between evaluations
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn eval_interval(mut self, v: usize) -> Self {
        self.eval_interval = v;
        self
    }

    /// (Deprecated) Sets the evaluation threshold.
    ///
    /// This method is currently unimplemented and may be removed in future versions.
    pub fn eval_threshold(/*mut */ self, _v: f32) -> Self {
        unimplemented!();
        // self.eval_threshold = Some(v);
        // self
    }

    /// Sets the interval between optimization updates.
    ///
    /// # Arguments
    ///
    /// * `opt_interval` - Number of environment steps between optimizations
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn opt_interval(mut self, opt_interval: usize) -> Self {
        self.opt_interval = opt_interval;
        self
    }

    /// Sets the interval for flushing recorded metrics.
    ///
    /// # Arguments
    ///
    /// * `flush_record_interval` - Number of optimization steps between flushes
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn flush_record_interval(mut self, flush_record_interval: usize) -> Self {
        self.flush_record_interval = flush_record_interval;
        self
    }

    /// Sets the interval for recording computational performance metrics.
    ///
    /// # Arguments
    ///
    /// * `record_compute_cost_interval` - Number of optimization steps between recordings
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn record_compute_cost_interval(mut self, record_compute_cost_interval: usize) -> Self {
        self.record_compute_cost_interval = record_compute_cost_interval;
        self
    }

    /// Sets the interval for recording agent-specific information.
    ///
    /// # Arguments
    ///
    /// * `record_agent_info_interval` - Number of optimization steps between recordings
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn record_agent_info_interval(mut self, record_agent_info_interval: usize) -> Self {
        self.record_agent_info_interval = record_agent_info_interval;
        self
    }

    /// Sets the initial warmup period before optimization begins.
    ///
    /// # Arguments
    ///
    /// * `warmup_period` - Number of environment steps in the warmup period
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn warmup_period(mut self, warmup_period: usize) -> Self {
        self.warmup_period = warmup_period;
        self
    }

    /// Sets the interval for saving model checkpoints.
    ///
    /// # Arguments
    ///
    /// * `save_interval` - Number of optimization steps between checkpoints
    ///
    /// # Returns
    ///
    /// Self with the updated configuration
    pub fn save_interval(mut self, save_interval: usize) -> Self {
        self.save_interval = save_interval;
        self
    }

    /// Loads configuration from a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// Result containing the loaded configuration
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

    /// Saves the configuration to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the configuration will be saved
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tempdir::TempDir;

//     #[test]
//     fn test_serde_trainer_builder() -> Result<()> {
//         let builder = TrainerBuilder::default()
//             .max_opts(100)
//             .eval_interval(10000)
//             .n_episodes_per_eval(5)
//             .model_dir("some/directory");

//         let dir = TempDir::new("trainer_builder")?;
//         let path = dir.path().join("trainer_builder.yaml");
//         println!("{:?}", path);

//         builder.save(&path)?;
//         let builder_ = TrainerBuilder::load(&path)?;
//         assert_eq!(builder, builder_);
//         // let yaml = serde_yaml::to_string(&trainer)?;
//         // println!("{}", yaml);
//         // assert_eq!(
//         //     yaml,
//         //     "---\n\
//         //      max_opts: 100\n\
//         //      eval_interval: 10000\n\
//         //      n_episodes_per_eval: 5\n\
//         //      eval_threshold: ~\n\
//         //      model_dir: some/directory\n\
//         // "
//         // );
//         Ok(())
//     }
// }
