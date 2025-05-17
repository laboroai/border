//! Recorder trait for handling record storage and model persistence.
//!
//! This module defines the [`Recorder`] trait, which provides an interface for
//! recording training metrics, storing records, and managing model persistence.
//! Implementations of this trait can handle different storage backends and
//! recording strategies.

use super::Record;
use crate::{Agent, Env, ReplayBufferBase};
use anyhow::Result;
use std::path::Path;

/// A trait for recording training metrics and managing model persistence.
///
/// The `Recorder` trait defines an interface for handling various aspects of
/// reinforcement learning training, including:
///
/// * Recording training metrics and observations
/// * Storing and aggregating records
/// * Managing model checkpoints
///
/// # Type Parameters
///
/// * `E` - The environment type that implements the [`Env`] trait
/// * `R` - The replay buffer type that implements the [`ReplayBufferBase`] trait
pub trait Recorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Writes a record to the recorder's output destination.
    ///
    /// This method is called for each record that needs to be written immediately,
    /// such as during training or evaluation.
    ///
    /// # Arguments
    ///
    /// * `record` - The record to be written
    fn write(&mut self, record: Record);

    /// Stores a record for later processing or aggregation.
    ///
    /// This method is used to collect records that will be processed together,
    /// typically during a flush operation.
    ///
    /// # Arguments
    ///
    /// * `record` - The record to be stored
    fn store(&mut self, record: Record);

    /// Writes aggregated values from stored records.
    ///
    /// This method is called to process and write any accumulated records,
    /// typically at the end of an episode or training step.
    ///
    /// # Arguments
    ///
    /// * `step` - The current training step or episode number
    fn flush(&mut self, step: i64);

    /// Saves the current state of the agent's model.
    ///
    /// This method is used to create checkpoints of the agent's model during training.
    /// The implementation should handle the specific requirements of the agent's
    /// model format and storage needs.
    ///
    /// # Arguments
    ///
    /// * `base` - The base path where the model should be saved
    /// * `agent` - The agent whose model should be saved
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the save operation
    ///
    /// # Note
    ///
    /// The default implementation is unimplemented and will panic if called.
    /// Implementations should override this method to provide model saving functionality.
    #[allow(unused_variables)]
    fn save_model(&self, base: &Path, agent: &Box<dyn Agent<E, R>>) -> Result<()> {
        unimplemented!();
    }

    /// Loads a previously saved model state into the agent.
    ///
    /// This method is used to restore an agent's model from a checkpoint.
    /// The implementation should handle the specific requirements of the agent's
    /// model format and storage needs.
    ///
    /// # Arguments
    ///
    /// * `base` - The base path where the model was saved
    /// * `agent` - The agent whose model should be loaded
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the load operation
    ///
    /// # Note
    ///
    /// The default implementation is unimplemented and will panic if called.
    /// Implementations should override this method to provide model loading functionality.
    #[allow(unused_variables)]
    fn load_model(&self, base: &Path, agent: &mut Box<dyn Agent<E, R>>) -> Result<()> {
        unimplemented!();
    }
}
