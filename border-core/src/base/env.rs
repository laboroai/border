//! Environment.
use super::{Act, Info, Obs, Step};
use crate::record::Record;
use anyhow::Result;

/// Represents an environment, typically an MDP.
pub trait Env {
    /// Configurations.
    type Config;

    /// Observation of the environment.
    type Obs: Obs;

    /// Action of the environment.
    type Act: Act;

    /// Information in the [self::Step] object.
    type Info: Info;

    /// Builds an environment.
    fn build(config: &Self::Config, seed: i64) -> Result<Self>
    where
        Self: Sized;

    /// Performes an environment step.
    fn step(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized;

    /// Resets the i-th environment if `is_done[i] == 1`.
    /// The i-th return value should be ignored if `is_done[i] == 0`.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs>;

    /// Performes an environment step and reset the environment if an episode ends.
    fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized;
}
