//! Evaluate [`Agent`].
use crate::{Agent, Env, ReplayBufferBase, record::Record};
use anyhow::Result;
mod default_evaluator;
pub use default_evaluator::DefaultEvaluator;

/// Evaluate [`Agent`].
pub trait Evaluator<E: Env> {
    /// Evaluate [`Agent`].
    ///
    /// The caller of this method needs to handle the internal state of `agent`,
    /// like training/evaluation mode.
    fn evaluate<R>(&mut self, agent: &mut Box<dyn Agent<E, R>>) -> Result<Record>
    where
        R: ReplayBufferBase;
}
