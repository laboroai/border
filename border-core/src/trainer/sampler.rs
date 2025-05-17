//! Experience sampling and replay buffer management.
//!
//! This module provides functionality for sampling experiences from the environment
//! and storing them in a replay buffer. It handles the interaction between the agent,
//! environment, and replay buffer, while also tracking performance metrics.
//!
//! # Sampling Process
//!
//! The sampling process involves:
//!
//! 1. Environment Interaction:
//!    * Agent observes environment state
//!    * Agent selects and executes action
//!    * Environment transitions to new state
//!
//! 2. Experience Processing:
//!    * Convert environment step into transition
//!    * Store transition in replay buffer
//!    * Track episode length and performance metrics
//!
//! 3. Performance Monitoring:
//!    * Monitor episode length
//!    * Record environment metrics
use crate::{record::Record, Agent, Env, ExperienceBufferBase, ReplayBufferBase, StepProcessor};
use anyhow::Result;

/// Manages the sampling of experiences from the environment.
///
/// This struct handles the interaction between the agent and environment,
/// processes the resulting experiences, and stores them in a replay buffer.
/// It also tracks various metrics about the sampling process.
///
/// # Type Parameters
///
/// * `E` - The environment type
/// * `P` - The step processor type
pub struct Sampler<E, P>
where
    E: Env,
    P: StepProcessor<E>,
{
    /// The environment being sampled from
    env: E,

    /// Previous observation from the environment
    prev_obs: Option<E::Obs>,

    /// Processor for converting steps into transitions
    step_processor: P,
}

impl<E, P> Sampler<E, P>
where
    E: Env,
    P: StepProcessor<E>,
{
    /// Creates a new sampler with the given environment and step processor.
    ///
    /// # Arguments
    ///
    /// * `env` - The environment to sample from
    /// * `step_processor` - The processor for converting steps into transitions
    ///
    /// # Returns
    ///
    /// A new `Sampler` instance
    pub fn new(env: E, step_processor: P) -> Self {
        Self {
            env,
            prev_obs: None,
            step_processor,
        }
    }

    /// Samples an experience and pushes it to the replay buffer.
    ///
    /// This method:
    /// 1. Resets the environment if needed
    /// 2. Samples an action from the agent
    /// 3. Applies the action to the environment
    /// 4. Processes the resulting step
    /// 5. Stores the experience in the replay buffer
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to sample actions from
    /// * `buffer` - The replay buffer to store experiences in
    ///
    /// # Returns
    ///
    /// A `Record` containing metrics about the sampling process
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The environment fails to reset
    /// * The environment step fails
    /// * The replay buffer operation fails
    pub fn sample_and_push<R, R_>(
        &mut self,
        agent: &mut Box<dyn Agent<E, R>>,
        buffer: &mut R_,
    ) -> Result<Record>
    where
        R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
        R_: ExperienceBufferBase<Item = R::Item>,
    {
        // Reset environment(s) if required
        if self.prev_obs.is_none() {
            // For a vectorized environments, reset all environments in `env`
            // by giving `None` to reset() method
            self.prev_obs = Some(self.env.reset(None)?);
            self.step_processor
                .reset(self.prev_obs.as_ref().unwrap().clone());
        }

        // Sample an action and apply it to the environment
        let (step, record, is_done) = {
            let act = agent.sample(self.prev_obs.as_ref().unwrap());
            let (step, record) = self.env.step_with_reset(&act);
            let is_done = step.is_done(); // not support vectorized env
            (step, record, is_done)
        };

        // Update previouos observation
        self.prev_obs = match is_done {
            true => Some(step.init_obs.clone().expect("Failed to unwrap init_obs")),
            false => Some(step.obs.clone()),
        };

        // Produce transition
        let transition = self.step_processor.process(step);

        // Push transition
        buffer.push(transition)?;

        // Reset step processor
        if is_done {
            self.step_processor
                .reset(self.prev_obs.as_ref().unwrap().clone());
        }

        Ok(record)
    }
}
