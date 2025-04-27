//! Environment interface for reinforcement learning.
//!
//! This module defines the core interface for environments in reinforcement learning.
//! An environment represents a Markov Decision Process (MDP) where an agent can interact
//! through actions and receive observations and rewards in return.

/// Represents a reinforcement learning environment, typically modeled as a Markov Decision Process (MDP).
///
/// This trait defines the interface for environments in reinforcement learning. It provides methods for:
/// - Building the environment with specific configurations
/// - Performing steps in the environment
/// - Resetting the environment to its initial state
/// - Handling episode termination and truncation
///
/// # Associated Types
///
/// * `Config` - Configuration parameters for the environment
/// * `Obs` - The type of observations returned by the environment
/// * `Act` - The type of actions accepted by the environment
/// * `Info` - Additional information returned with each step
///
/// # Examples
///
/// A typical interaction with an environment might look like:
/// ```ignore
/// let config = EnvConfig::default();
/// let mut env = Env::build(&config, 42)?;
/// let mut obs = env.reset(None)?;
///
/// loop {
///     let action = agent.sample(&obs);
///     let (step, _) = env.step(&action);
///     obs = step.obs;
///
///     if step.is_done() {
///         break;
///     }
/// }
/// ```
use super::{Act, Info, Obs, Step};
use crate::record::Record;
use anyhow::Result;

/// Environment interface for reinforcement learning.
pub trait Env {
    /// Configuration parameters for the environment.
    ///
    /// This type should contain all necessary parameters to build and configure
    /// the environment, such as environment-specific settings, rendering options,
    /// or difficulty levels.
    type Config: Clone;

    /// The type of observations returned by the environment.
    ///
    /// Observations represent the state of the environment as perceived by the agent.
    /// This type must implement the [`Obs`] trait.
    type Obs: Obs;

    /// The type of actions accepted by the environment.
    ///
    /// Actions represent the decisions made by the agent that affect the environment.
    /// This type must implement the [`Act`] trait.
    type Act: Act;

    /// Additional information returned with each step.
    ///
    /// This type can be used to provide extra information about the environment's state
    /// that isn't part of the observation. It must implement the [`Info`] trait.
    type Info: Info;

    /// Builds a new instance of the environment with the given configuration and random seed.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the environment
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// A new instance of the environment or an error if construction fails
    fn build(config: &Self::Config, seed: i64) -> Result<Self>
    where
        Self: Sized;

    /// Performs a single step in the environment.
    ///
    /// This method advances the environment by one time step, applying the given action
    /// and returning the resulting observation, reward, and termination information.
    ///
    /// # Arguments
    ///
    /// * `a` - The action to apply to the environment
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// 1. A [`Step`] object with the next observation, reward, and termination info
    /// 2. A [`Record`] with additional environment-specific information
    fn step(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized;

    /// Resets the environment to its initial state.
    ///
    /// This method resets the environment when:
    /// - `is_done` is `None` (initial reset)
    /// - `is_done[0] == 1` (episode termination)
    ///
    /// # Arguments
    ///
    /// * `is_done` - Optional vector indicating which environments to reset
    ///
    /// # Note
    ///
    /// While the interface supports vectorized environments through `is_done`,
    /// the current implementation only supports single environments.
    /// Therefore, `is_done.len()` is expected to be 1.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs>;

    /// Performs a step and automatically resets the environment if the episode ends.
    ///
    /// This is a convenience method that combines [`step`] and [`reset`] operations.
    /// If the step results in episode termination, the environment is automatically
    /// reset and the initial observation is included in the returned step.
    ///
    /// # Arguments
    ///
    /// * `a` - The action to apply to the environment
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// 1. A [`Step`] object with the next observation, reward, and termination info
    /// 2. A [`Record`] with additional environment-specific information
    ///
    /// [`step`]: Env::step
    /// [`reset`]: Env::reset
    fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized,
    {
        let (step, record) = self.step(a);
        assert_eq!(step.is_terminated.len(), 1);
        let step = if step.is_done() {
            let init_obs = self.reset(None).unwrap();
            Step {
                act: step.act,
                obs: step.obs,
                reward: step.reward,
                is_terminated: step.is_terminated,
                is_truncated: step.is_truncated,
                info: step.info,
                init_obs: Some(init_obs),
            }
        } else {
            step
        };

        (step, record)
    }

    /// Resets the environment with a specific index.
    ///
    /// This method is primarily used during evaluation to control the initial state
    /// of the environment. The index can be used in various ways, such as:
    /// - As a random seed for deterministic initialization
    /// - To select specific starting conditions
    /// - To control the difficulty level
    ///
    /// # Arguments
    ///
    /// * `ix` - An index used to control the reset behavior
    ///
    /// # Note
    ///
    /// This method is called by the [`Trainer`] during evaluation to ensure
    /// consistent testing conditions.
    ///
    /// [`Trainer`]: crate::Trainer
    fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs>;
}
