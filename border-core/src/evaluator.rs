//! Evaluation interface for reinforcement learning agents.
//!
//! This module provides interfaces and implementations for evaluating the performance
//! of reinforcement learning agents. Evaluation is a crucial step in the development
//! of reinforcement learning systems, allowing developers to:
//! - Measure the effectiveness of trained agents
//! - Compare different algorithms or hyperparameters
//! - Monitor training progress
//! - Validate the generalization of learned policies

use crate::{record::Record, Agent, Env, ReplayBufferBase};
use anyhow::Result;
mod default_evaluator;
pub use default_evaluator::DefaultEvaluator;

/// Interface for evaluating reinforcement learning agents.
///
/// This trait defines the standard interface for evaluating agents in different
/// environments. Implementations of this trait should:
/// - Run the agent in the environment for a specified number of episodes
/// - Collect performance metrics (e.g., average return, success rate)
/// - Return the results in a standardized format
///
/// # Type Parameters
///
/// * `E` - The environment type that the agent operates in
///
/// # Examples
///
/// ```ignore
/// struct CustomEvaluator<E: Env> {
///     env: E,
///     n_episodes: usize,
/// }
///
/// impl<E: Env> Evaluator<E> for CustomEvaluator<E> {
///     fn evaluate<R>(&mut self, agent: &mut Box<dyn Agent<E, R>>) -> Result<Record>
///     where
///         R: ReplayBufferBase,
///     {
///         // Custom evaluation logic
///         // ...
///     }
/// }
/// ```
pub trait Evaluator<E: Env> {
    /// Evaluates an agent's performance in the environment.
    ///
    /// This method should:
    /// 1. Set the agent to evaluation mode
    /// 2. Run the agent in the environment for the specified number of episodes
    /// 3. Collect and aggregate performance metrics
    /// 4. Return the results in a [`Record`]
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to evaluate
    ///
    /// # Returns
    ///
    /// A tuple of (performance metric, [`Record`] containing the evaluation results)
    /// 
    /// In [`Trainer`], the performance metric is used to choose the best model.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The environment fails to reset or step
    /// - The agent fails to produce actions
    /// - The evaluation process encounters an unexpected error
    ///
    /// # Note
    ///
    /// The caller is responsible for managing the agent's internal state,
    /// such as switching between training and evaluation modes. This allows
    /// for flexible evaluation strategies that may require different agent
    /// configurations.
    /// 
    /// [`Trainer`]: crate::Trainer
    fn evaluate<R>(&mut self, agent: &mut Box<dyn Agent<E, R>>) -> Result<(f32, Record)>
    where
        R: ReplayBufferBase;
}
