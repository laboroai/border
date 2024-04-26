//! Policy.
use super::Env;
// use anyhow::Result;

/// A policy on an environment.
///
/// Policy is a mapping from an observation to an action.
/// The mapping can be either of deterministic or stochastic.
pub trait Policy<E: Env> {
    /// Sample an action given an observation.
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}

/// A configurable object, having type parameter.
pub trait Configurable<E: Env> {
    /// Configuration.
    type Config: Clone;

    /// Builds the object.
    fn build(config: Self::Config) -> Self;
}
