//! Policy.
use super::Env;
// use anyhow::Result;

/// Represents a policy. on an environment. It is based on a mapping from an observation
/// to an action. The mapping can be either of deterministic or stochastic.
pub trait Policy<E: Env> {
    /// Configuration of the policy.
    type Config: Clone;

    /// Builds the policy.
    fn build(config: Self::Config) -> Self;

    /// Sample an action given an observation.
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}
