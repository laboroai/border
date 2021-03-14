use std::{fmt::Debug, path::Path, error::Error};
use crate::core::record::Record;

/// Represents an observation of the environment.
pub trait Obs: Clone + Debug {
    /// Returns a dummy observation.
    ///
    /// The observation created with this method is ignored.
    fn dummy(n_procs: usize) -> Self;

    /// Replace elements of observation where `is_done[i] == 1.0`.
    /// This method assumes that `is_done.len() == n_procs`.
    fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self;
}

/// Represents an action of the environment.
pub trait Act: Clone + Debug {}

/// Represents additional information to `Obs` and `Act`.
pub trait Info {}

/// Represents all information given at every step of agent-envieronment interaction.
/// `reward` and `is_done` have the same length, the number of processes (environments).
pub struct Step<E: Env> {
    pub act: E::Act,
    pub obs: E::Obs,
    pub reward: Vec<f32>,
    pub is_done: Vec<i8>,
    pub info: E::Info,
}

impl<E: Env> Step<E> {
    pub fn new(obs: E::Obs, act: E::Act, reward: Vec<f32>, is_done: Vec<i8>, info: E::Info) -> Self {
        Step {
            act,
            obs,
            reward,
            is_done,
            info,
        }
    }
}

/// Represents an environment, typically an MDP.
pub trait Env {
    type Obs: Obs;
    type Act: Act;
    type Info: Info;

    fn step(&mut self, a: &Self::Act) -> (Step<Self>, Record) where Self: Sized;

    /// Reset the i-th environment if `is_done[i]==1.0`.
    /// Thei-th return value should be ignored if `is_done[i]==0.0`.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs, Box<dyn Error>>;
}

/// Represents a policy. on an environment. It is based on a mapping from an observation
/// to an action. The mapping can be either of deterministic or stochastic.
pub trait Policy<E: Env> {
    /// Sample an action given an observation.
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}

/// Represents a trainable policy on an environment.
pub trait Agent<E: Env>: Policy<E> {
    /// Set the policy to training mode.
    fn train(&mut self);

    /// Set the policy to evaluation mode.
    fn eval(&mut self);

    /// Return if it is in training mode.
    fn is_train(&self) -> bool;

    /// Observe a [crate::core::base::Step] object.
    /// The agent is expected to do training its policy based on the observation.
    ///
    /// If an optimization step was performed, it returns `Some(crate::core::record::Record)`,
    /// otherwise `None`.
    fn observe(&mut self, step: Step<E>) -> Option<Record>;

    /// Push observation to the agent.
    /// This method is used when resetting the environment.
    fn push_obs(&self, obs: &E::Obs);

    /// Save the agent in the given directory.
    /// This method commonly creates a number of files consisting the agent
    /// into the given directory. For example, [crate::agents::dqn::DQN] agent saves
    /// two Q-networks corresponding to the original and target networks.
    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>>;

    /// Load the agent from the given directory.
    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>>;
}
