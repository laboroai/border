//! A trainable policy that can interact with an environment and learn from experience.
//!
//! The [`Agent`] trait extends [`Policy`] with training capabilities, allowing the policy to
//! learn from interactions with the environment. It provides methods for training, evaluation,
//! parameter optimization, and model persistence.
use super::{Env, Policy, ReplayBufferBase};
use crate::record::Record;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// A trainable policy that can learn from environment interactions.
///
/// This trait extends [`Policy`] with training capabilities, allowing the policy to:
/// - Switch between training and evaluation modes
/// - Perform optimization steps using experience from a replay buffer
/// - Save and load model parameters
///
/// The agent operates in two distinct modes:
/// - Training mode: The policy may be stochastic to facilitate exploration
/// - Evaluation mode: The policy is typically deterministic for consistent performance
///
/// During training, the agent uses a replay buffer to store and sample experiences,
/// which are then used to update the policy's parameters through optimization steps.
pub trait Agent<E: Env, R: ReplayBufferBase>: Policy<E> {
    /// Switches the agent to training mode.
    ///
    /// In training mode, the policy may become stochastic to facilitate exploration.
    /// This is typically implemented by enabling noise or randomness in the action selection process.
    fn train(&mut self) {
        unimplemented!();
    }

    /// Switches the agent to evaluation mode.
    ///
    /// In evaluation mode, the policy typically becomes deterministic to ensure
    /// consistent performance. This is often implemented by disabling noise or
    /// using the mean action instead of sampling from a distribution.
    fn eval(&mut self) {
        unimplemented!();
    }

    /// Returns whether the agent is currently in training mode.
    ///
    /// This method is used to determine the agent's current mode and can be used
    /// to conditionally enable or disable certain behaviors.
    fn is_train(&self) -> bool {
        unimplemented!();
    }

    /// Performs a single optimization step using experiences from the replay buffer.
    ///
    /// This method updates the agent's parameters using a batch of transitions
    /// sampled from the provided replay buffer. The specific optimization algorithm
    /// (e.g., Q-learning, policy gradient) is determined by the agent's implementation.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The replay buffer containing experiences used for training
    fn opt(&mut self, buffer: &mut R) {
        let _ = self.opt_with_record(buffer);
    }

    /// Performs an optimization step and returns training metrics.
    ///
    /// Similar to [`opt`], but also returns a [`Record`] containing training metrics
    /// such as loss values, gradients, or other relevant statistics.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The replay buffer containing experiences used for training
    ///
    /// # Returns
    ///
    /// A [`Record`] containing training metrics and statistics
    ///
    /// [`opt`]: Agent::opt
    /// [`Record`]: crate::record::Record
    #[allow(unused_variables)]
    fn opt_with_record(&mut self, buffer: &mut R) -> Record {
        unimplemented!();
    }

    /// Saves the agent's parameters to the specified directory.
    ///
    /// This method serializes the agent's current state (e.g., neural network weights,
    /// policy parameters) to files in the given directory. The specific format and
    /// number of files created depends on the agent's implementation.
    ///
    /// # Arguments
    ///
    /// * `path` - The directory where parameters will be saved
    ///
    /// # Returns
    ///
    /// A vector of paths to the saved parameter files
    ///
    /// # Examples
    ///
    /// For example, a DQN agent might save two Q-networks (original and target)
    /// in separate files within the specified directory.
    #[allow(unused_variables)]
    fn save_params(&self, path: &Path) -> Result<Vec<PathBuf>> {
        unimplemented!();
    }

    /// Loads the agent's parameters from the specified directory.
    ///
    /// This method deserializes the agent's state from files in the given directory,
    /// restoring the agent to a previously saved state.
    ///
    /// # Arguments
    ///
    /// * `path` - The directory containing the saved parameter files
    #[allow(unused_variables)]
    fn load_params(&mut self, path: &Path) -> Result<()> {
        unimplemented!();
    }

    /// Returns a reference to the agent as a type-erased `Any` value.
    ///
    /// This method is required for asynchronous training, allowing the agent to be
    /// stored in a type-erased container. The returned reference can be downcast
    /// to the concrete agent type when needed.
    fn as_any_ref(&self) -> &dyn std::any::Any {
        unimplemented!("as_any_ref() must be implemented for train_async()");
    }

    /// Returns a mutable reference to the agent as a type-erased `Any` value.
    ///
    /// This method is required for asynchronous training, allowing the agent to be
    /// stored in a type-erased container. The returned reference can be downcast
    /// to the concrete agent type when needed.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        unimplemented!("as_any_mut() must be implemented for train_async()");
    }
}
