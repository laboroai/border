//! Evaluate [`Policy`](crate::Policy).
use crate::{Env, Policy};
use anyhow::Result;
mod default_evaluator;
pub use default_evaluator::DefaultEvaluator;

/// Evaluate [`Policy`](crate::Policy).
pub trait Evaluator<E: Env, P: Policy<E>> {
    /// Evaluate [`Policy`](crate::Policy).
    ///
    /// The caller of this method needs to handle the internal state of `policy`,
    /// like training/evaluation mode.
    fn evaluate(&mut self, policy: &mut P) -> Result<f32>;
}
