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

    /// Do an optimization step.
    fn opt(&mut self, buffer: &mut R) -> Option<Record>;

    /// Save the agent in the given directory.
    /// This method commonly creates a number of files consisting the agent
    /// in the directory. For example, the DQN agent in `border_tch_agent` crate saves
    /// two Q-networks corresponding to the original and target networks.
    fn save<T: AsRef<Path>>(&self, path: T) -> Result<()>;

    /// Load the agent from the given directory.
    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()>;
}
