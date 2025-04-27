//! Default implementation of the [`Evaluator`] trait.
//!
//! This module provides a simple evaluator that runs a fixed number of episodes
//! and calculates the average return across all episodes.

use super::Evaluator;
use crate::{record::Record, Agent, Env, ReplayBufferBase};
use anyhow::Result;

/// A default implementation of the [`Evaluator`] trait.
///
/// This evaluator runs a specified number of episodes and calculates the average
/// return (cumulative reward) across all episodes. It is useful for:
/// - Evaluating the performance of trained agents
/// - Comparing different policies or algorithms
/// - Monitoring training progress
///
/// # Type Parameters
///
/// * `E` - The environment type
///
/// # Examples
///
/// ```ignore
/// let config = EnvConfig::default();
/// let mut evaluator = DefaultEvaluator::new(&config, 42, 10)?;
///
/// // Evaluate a policy
/// let record = evaluator.evaluate(&mut agent)?;
/// println!("Average return: {}", record.get_scalar("Episode return")?);
/// ```
pub struct DefaultEvaluator<E: Env> {
    /// The number of episodes to run during evaluation.
    n_episodes: usize,

    /// The environment instance used for evaluation.
    env: E,
}

impl<E: Env> Evaluator<E> for DefaultEvaluator<E> {
    /// Evaluates a policy by running multiple episodes and calculating the average return.
    ///
    /// This method:
    /// 1. Runs the specified number of episodes
    /// 2. For each episode:
    ///    - Resets the environment with a unique index
    ///    - Runs the episode until termination
    ///    - Accumulates the total reward
    /// 3. Returns the average return across all episodes
    ///
    /// # Arguments
    ///
    /// * `policy` - The policy to evaluate
    ///
    /// # Returns
    ///
    /// A [`Record`] containing the average return across all episodes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The environment fails to reset
    /// - The environment fails to step
    fn evaluate<R>(&mut self, policy: &mut Box<dyn Agent<E, R>>) -> Result<Record>
    where
        R: ReplayBufferBase,
    {
        let mut r_total = 0f32;

        for ix in 0..self.n_episodes {
            let mut prev_obs = self.env.reset_with_index(ix)?;

            loop {
                let act = policy.sample(&prev_obs);
                let (step, _) = self.env.step(&act);
                r_total += step.reward[0];
                if step.is_done() {
                    break;
                }
                prev_obs = step.obs;
            }
        }

        let name = format!("Episode return");
        Ok(Record::from_scalar(name, r_total / self.n_episodes as f32))
    }
}

impl<E: Env> DefaultEvaluator<E> {
    /// Constructs a new [`DefaultEvaluator`].
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the environment
    /// * `seed` - Random seed for environment initialization
    /// * `n_episodes` - Number of episodes to run during evaluation
    ///
    /// # Returns
    ///
    /// A new evaluator instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = EnvConfig::default();
    /// let evaluator = DefaultEvaluator::new(&config, 42, 10)?;
    /// ```
    pub fn new(config: &E::Config, seed: i64, n_episodes: usize) -> Result<Self> {
        Ok(Self {
            n_episodes,
            env: E::build(config, seed)?,
        })
    }
}
