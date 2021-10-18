//! Policy.
use super::Env;

/// Represents a policy. on an environment. It is based on a mapping from an observation
/// to an action. The mapping can be either of deterministic or stochastic.
pub trait Policy<E: Env> {
    /// Sample an action given an observation.
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}
