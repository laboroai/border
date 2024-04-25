//! Policy.
use super::Env;
// use anyhow::Result;

/// A policy on an environment.
///
/// Policy is a mapping from an observation to an action.
/// The mapping can be either of deterministic or stochastic.
pub trait Policy_<E: Env> {
    /// Sample an action given an observation.
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}

/// A configurable policy.
pub trait Policy<E: Env> {
    /// Configuration of the policy.
    type Config: Clone;

    /// Builds the policy.
    fn build(config: Self::Config) -> Self;
}
