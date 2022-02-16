//! Environment.
use super::{Act, Info, Obs, Step};
use crate::record::Record;
use anyhow::Result;

/// Represents an environment, typically an MDP.
pub trait Env {
    /// Configurations.
    type Config: Clone;

    /// Observation of the environment.
    type Obs: Obs;

    /// Action of the environment.
    type Act: Act;

    /// Information in the [self::Step] object.
    type Info: Info;

    /// Builds an environment with a given random seed.
    fn build(config: &Self::Config, seed: i64) -> Result<Self>
    where
        Self: Sized;

    /// Performes an environment step.
    fn step(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized;

    /// Resets the environment if `is_done[0] == 1`. or `is_done.is_none()`.
    ///
    /// Old versions of the library supports vectorized environments and `is_done` was
    /// used to reset a part of the vectorized environments. Currenly, vecorized environment
    /// is not supported and `is_done.len()` is expected to be 1.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs>;

    /// Performes an environment step and reset the environment if an episode ends.
    fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized;

    /// Resets the environment with a given index.
    ///
    /// The index is used in an arbitrary way. For example, it can be used as a random seed,
    /// which is useful when evaluation of a trained agent. Actually, this method is called
    /// in [`Trainer`] for evaluation. This method does not support vectorized environments.
    ///
    /// [`Trainer`]: crate::Trainer
    fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs>;
}
