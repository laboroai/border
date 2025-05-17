//! Step processing interface for reinforcement learning.
//!
//! This module defines the core interfaces for processing environment steps in reinforcement learning.
//! It provides structures and traits for handling the transition between states, including
//! observations, actions, rewards, and episode termination information.

use super::Env;

/// Additional information that can be associated with environment steps.
///
/// This trait is used to define custom information types that can be attached to
/// environment steps. It is typically implemented for types that provide extra
/// context about the environment's state or the agent's actions.
///
/// # Examples
///
/// ```ignore
/// #[derive(Debug)]
/// struct CustomInfo {
///     velocity: f32,
///     position: (f32, f32),
/// }
///
/// impl Info for CustomInfo {}
/// ```
pub trait Info {}

impl Info for () {}

/// Represents a single step in the environment, containing the action taken,
/// the resulting observation, reward, and episode status.
///
/// This struct encapsulates all the information produced by an environment
/// during a single interaction step. It is used to create transitions of the form
/// `(o_t, a_t, o_t+1, r_t)` for training reinforcement learning agents.
///
/// # Type Parameters
///
/// * `E` - The environment type that produced this step
///
/// # Fields
///
/// * `act` - The action taken by the agent
/// * `obs` - The observation received from the environment
/// * `reward` - The reward received for the action
/// * `is_terminated` - Flags indicating if the episode has terminated
/// * `is_truncated` - Flags indicating if the episode has been truncated
/// * `info` - Additional environment-specific information
/// * `init_obs` - The initial observation of the next episode (if applicable)
///
/// # Examples
///
/// ```ignore
/// let step = Step::new(
///     observation,
///     action,
///     vec![0.5],  // reward
///     vec![0],    // not terminated
///     vec![0],    // not truncated
///     info,
///     None,       // no initial observation
/// );
///
/// if step.is_done() {
///     // Handle episode completion
/// }
/// ```
pub struct Step<E: Env> {
    /// The action taken by the agent in this step.
    pub act: E::Act,

    /// The observation received from the environment after taking the action.
    pub obs: E::Obs,

    /// The reward received for taking the action.
    pub reward: Vec<f32>,

    /// Flags indicating if the episode has terminated.
    /// A value of 1 indicates termination.
    pub is_terminated: Vec<i8>,

    /// Flags indicating if the episode has been truncated.
    /// A value of 1 indicates truncation.
    pub is_truncated: Vec<i8>,

    /// Additional environment-specific information.
    pub info: E::Info,

    /// The initial observation of the next episode, if applicable.
    /// This is used when an episode ends and a new one begins.
    pub init_obs: Option<E::Obs>,
}

impl<E: Env> Step<E> {
    /// Constructs a new [`Step`] object with the given components.
    ///
    /// # Arguments
    ///
    /// * `obs` - The observation received from the environment
    /// * `act` - The action taken by the agent
    /// * `reward` - The reward received for the action
    /// * `is_terminated` - Flags indicating episode termination
    /// * `is_truncated` - Flags indicating episode truncation
    /// * `info` - Additional environment-specific information
    /// * `init_obs` - The initial observation of the next episode
    ///
    /// # Returns
    ///
    /// A new [`Step`] object containing all the provided information
    pub fn new(
        obs: E::Obs,
        act: E::Act,
        reward: Vec<f32>,
        is_terminated: Vec<i8>,
        is_truncated: Vec<i8>,
        info: E::Info,
        init_obs: Option<E::Obs>,
    ) -> Self {
        Step {
            act,
            obs,
            reward,
            is_terminated,
            is_truncated,
            info,
            init_obs,
        }
    }

    /// Checks if the episode has ended, either through termination or truncation.
    ///
    /// # Returns
    ///
    /// `true` if the episode has ended, `false` otherwise
    #[inline]
    pub fn is_done(&self) -> bool {
        self.is_terminated[0] == 1 || self.is_truncated[0] == 1
    }
}

/// Processes environment steps and produces items for a replay buffer.
///
/// This trait defines the interface for converting [`Step`] objects into items
/// that can be stored in a replay buffer. It is used by the [`Trainer`] to
/// transform environment interactions into training samples.
///
/// # Type Parameters
///
/// * `E` - The environment type
///
/// # Associated Types
///
/// * `Config` - Configuration parameters for the processor
/// * `Output` - The type of items produced by the processor
///
/// # Examples
///
/// ```ignore
/// struct SimpleProcessor;
///
/// impl<E: Env> StepProcessor<E> for SimpleProcessor {
///     type Config = ();
///     type Output = (E::Obs, E::Act, E::Obs, f32);
///
///     fn build(_: &Self::Config) -> Self {
///         Self
///     }
///
///     fn reset(&mut self, _: E::Obs) {}
///
///     fn process(&mut self, step: Step<E>) -> Self::Output {
///         (step.init_obs.unwrap(), step.act, step.obs, step.reward[0])
///     }
/// }
/// ```
///
/// [`Trainer`]: crate::Trainer
pub trait StepProcessor<E: Env> {
    /// Configuration parameters for the processor.
    ///
    /// This type must implement `Clone` to support building multiple instances
    /// with the same configuration.
    type Config: Clone;

    /// The type of items produced by the processor.
    ///
    /// This type should match the `Item` type of the replay buffer that will
    /// store the processed steps.
    type Output;

    /// Builds a new processor with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration parameters
    ///
    /// # Returns
    ///
    /// A new instance of the processor
    fn build(config: &Self::Config) -> Self;

    /// Resets the processor with a new initial observation.
    ///
    /// This method is called at the start of each episode to initialize
    /// the processor with the first observation.
    ///
    /// # Arguments
    ///
    /// * `init_obs` - The initial observation of the episode
    fn reset(&mut self, init_obs: E::Obs);

    /// Processes a step and produces an item for the replay buffer.
    ///
    /// This method transforms a [`Step`] object into an item that can be
    /// stored in a replay buffer. The transformation typically involves
    /// creating a transition tuple of the form `(o_t, a_t, o_t+1, r_t)`.
    ///
    /// # Arguments
    ///
    /// * `step` - The step to process
    ///
    /// # Returns
    ///
    /// An item ready to be stored in a replay buffer
    fn process(&mut self, step: Step<E>) -> Self::Output;
}
