//! Core interfaces and traits for reinforcement learning.
//!
//! This module provides the fundamental building blocks for reinforcement learning systems,
//! including interfaces for environments, agents, policies, and experience replay.
//! These interfaces define the core abstractions that enable the implementation of
//! various reinforcement learning algorithms.

mod agent;
mod batch;
mod env;
mod policy;
mod replay_buffer;
mod step;
pub use agent::Agent;
pub use batch::TransitionBatch;
pub use env::Env;
pub use policy::{Configurable, Policy};
pub use replay_buffer::{ExperienceBufferBase, NullReplayBuffer, ReplayBufferBase};
use std::fmt::Debug;
pub use step::{Info, Step, StepProcessor};

/// A trait representing observations from an environment.
///
/// This trait defines the interface for observations in reinforcement learning.
/// Observations represent the state of the environment as perceived by the agent.
///
/// # Requirements
///
/// Implementations must:
/// - Be cloneable for efficient copying
/// - Support debug formatting for logging and debugging
/// - Provide a method to determine the number of observations
///
/// # Note
///
/// While the interface supports vectorized environments through the `len` method,
/// the current implementation only supports single environments. Therefore,
/// `len()` is expected to return 1 in all cases.
///
/// # Examples
///
/// ```ignore
/// #[derive(Clone, Debug)]
/// struct SimpleObservation {
///     position: f32,
///     velocity: f32,
/// }
///
/// impl Obs for SimpleObservation {
///     fn len(&self) -> usize {
///         1  // Single observation
///     }
/// }
/// ```
pub trait Obs: Clone + Debug {
    /// Returns the number of observations in the object.
    ///
    /// # Returns
    ///
    /// The number of observations. Currently, this should always return 1
    /// as vectorized environments are not supported.
    fn len(&self) -> usize;
}

/// A trait representing actions that can be taken in an environment.
///
/// This trait defines the interface for actions in reinforcement learning.
/// Actions represent the decisions made by the agent that affect the environment.
///
/// # Requirements
///
/// Implementations must:
/// - Be cloneable for efficient copying
/// - Support debug formatting for logging and debugging
///
/// # Examples
///
/// ```ignore
/// #[derive(Clone, Debug)]
/// struct DiscreteAction {
///     action: usize,
///     num_actions: usize,
/// }
///
/// impl Act for DiscreteAction {
///     fn len(&self) -> usize {
///         self.num_actions
///     }
/// }
/// ```
pub trait Act: Clone + Debug {
    /// Returns the number of actions in the object.
    ///
    /// # Note
    ///
    /// This method is currently unimplemented and may be removed in future versions
    /// as it is not used in the current implementation.
    fn len(&self) -> usize {
        unimplemented!();
    }
}
