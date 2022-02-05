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
}
