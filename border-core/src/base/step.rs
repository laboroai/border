//! Environment step.
use super::Env;

/// Additional information to `Obs` and `Act`.
pub trait Info {}

/// All information given at every step of agent-envieronment interaction.
///
/// Old versions of the library support veectorized environments, which requires
/// elements in [`Step`] to be able to handle multiple values.
/// This is why `reward` and `is_done` are vector.
pub struct Step<E: Env> {
    /// Action.
    pub act: E::Act,

    /// Observation.
    pub obs: E::Obs,

    /// Reward.
    pub reward: Vec<f32>,

    /// Flag denoting if episode is done.
    pub is_done: Vec<i8>,

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
        is_done: Vec<i8>,
        info: E::Info,
        init_obs: E::Obs,
    ) -> Self {
        Step {
            act,
            obs,
            reward,
            is_done,
            info,
            init_obs,
        }
    }
}

/// Process [`Step`] and output an item [`Self::Output`].
///
/// This trait is used in [`Trainer`](crate::Trainer). [`Step`] object is transformed to
/// [`Self::Output`], which will be pushed into a replay buffer implementing
/// [`ReplayBufferBase`](crate::ReplayBufferBase). The type [`Self::Output`] must be the
/// same with [`ReplayBufferBase::PushedItem`].
///
/// [`Self::Output`]: StepProcessorBase::Output
/// [`ReplayBufferBase::PushedItem`]: crate::ReplayBufferBase::PushedItem
pub trait StepProcessorBase<E: Env> {
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
