//! Environment step.
use super::Env;

/// Additional information to `Obs` and `Act`.
pub trait Info {}

impl Info for () {}

/// Represents an action, observation and reward tuple `(a_t, o_t+1, r_t)`
/// with some additional information.
///
/// An environment emits [`Step`] object at every interaction steps.
/// This object might be used to create transitions `(o_t, a_t, o_t+1, r_t)`.
pub struct Step<E: Env> {
    /// Action.
    pub act: E::Act,

    /// Observation.
    pub obs: E::Obs,

    /// Reward.
    pub reward: Vec<f32>,

    /// Flag denoting if episode is terminated.
    pub is_terminated: Vec<i8>,

    /// Flag denoting if episode is truncated.
    pub is_truncated: Vec<i8>,

    /// Information defined by user.
    pub info: E::Info,

    /// Initial observation. If `is_done[i] == 0`, the corresponding element will not be used.
    pub init_obs: E::Obs,
}

impl<E: Env> Step<E> {
    /// Constructs a [`Step`] object.
    pub fn new(
        obs: E::Obs,
        act: E::Act,
        reward: Vec<f32>,
        is_terminated: Vec<i8>,
        is_truncated: Vec<i8>,
        info: E::Info,
        init_obs: E::Obs,
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

    #[inline]
    /// Terminated or truncated.
    pub fn is_done(&self) -> bool {
        self.is_terminated[0] == 1 || self.is_truncated[0] == 1
    }
}

/// Process [`Step`] and output an item [`Self::Output`].
///
/// This trait is used in [`Trainer`](crate::Trainer). [`Step`] object is transformed to
/// [`Self::Output`], which will be pushed into a replay buffer implementing
/// [`ExperienceBufferBase`](crate::ExperienceBufferBase).
/// The type [`Self::Output`] should be the same with [`ExperienceBufferBase::Item`].
///
/// [`Self::Output`]: StepProcessor::Output
/// [`ExperienceBufferBase::Item`]: crate::ExperienceBufferBase::Item
pub trait StepProcessor<E: Env> {
    /// Configuration.
    type Config: Clone;

    /// The type of transitions produced by this trait.
    type Output;

    /// Build a producer.
    fn build(config: &Self::Config) -> Self;

    /// Resets the object.
    fn reset(&mut self, init_obs: E::Obs);

    /// Processes a [`Step`] object.
    fn process(&mut self, step: Step<E>) -> Self::Output;
}
