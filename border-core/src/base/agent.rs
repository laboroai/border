//! Agent.
use super::{Env, Policy, ReplayBufferBase};
use crate::record::Record;
use anyhow::Result;
use std::path::Path;

/// Represents a trainable policy on an environment.
pub trait Agent<E: Env, R: ReplayBufferBase>: Policy<E> {
    /// Set the policy to training mode.
    fn train(&mut self);

    /// Set the policy to evaluation mode.
    fn eval(&mut self);

    /// Return if it is in training mode.
    fn is_train(&self) -> bool;

    /// Performs an optimization step.
    ///
    /// `buffer` is a replay buffer from which transitions will be taken
    /// for updating model parameters.
    fn opt(&mut self, buffer: &mut R) {
        let _ = self.opt_with_record(buffer);
    }

    /// Performs an optimization step and returns some information.
    fn opt_with_record(&mut self, buffer: &mut R) -> Record;

    /// Save the parameters of the agent in the given directory.
    /// This method commonly creates a number of files consisting the agent
    /// in the directory. For example, the DQN agent in `border_tch_agent` crate saves
    /// two Q-networks corresponding to the original and target networks.
    fn save_params(&self, path: &Path) -> Result<()>;

    /// Load the parameters of the agent from the given directory.
    fn load_params(&mut self, path: &Path) -> Result<()>;
}
