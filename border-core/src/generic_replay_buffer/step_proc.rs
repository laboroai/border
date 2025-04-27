//! Generic implementation of step processing for reinforcement learning.
//!
//! This module provides a generic implementation of the `StepProcessor` trait,
//! which handles the conversion of environment steps into transitions suitable
//! for training. It supports:
//! - 1-step TD backup for non-vectorized environments
//! - Generic observation and action types
//! - Efficient batch processing

use super::{BatchBase, GenericTransitionBatch};
use crate::{Env, Obs, StepProcessor};
use std::{default::Default, marker::PhantomData};

/// Configuration for the simple step processor.
#[derive(Clone, Debug)]
pub struct SimpleStepProcessorConfig {}

impl Default for SimpleStepProcessorConfig {
    /// Creates a new default configuration.
    fn default() -> Self {
        Self {}
    }
}

/// A generic implementation of the `StepProcessor` trait.
///
/// This processor converts environment steps into transitions suitable for
/// training reinforcement learning agents. It supports 1-step TD backup
/// for non-vectorized environments, meaning that each step contains exactly
/// one observation.
///
/// # Type Parameters
///
/// * `E` - The environment type, must implement `Env`
/// * `O` - The observation batch type, must implement `BatchBase` and `From<E::Obs>`
/// * `A` - The action batch type, must implement `BatchBase` and `From<E::Act>`
///
/// # Examples
///
/// ```rust
/// use border_core::{
///     Env, StepProcessor,
///     generic_replay_buffer::{SimpleStepProcessor, SimpleStepProcessorConfig}
/// };
///
/// let config = SimpleStepProcessorConfig::default();
/// let mut processor = SimpleStepProcessor::<MyEnv, Tensor, Tensor>::build(&config);
/// ```
pub struct SimpleStepProcessor<E, O, A> {
    /// The previous observation, used to construct transitions.
    prev_obs: Option<O>,
    /// Phantom data to hold the generic type parameters.
    phantom: PhantomData<(E, A)>,
}

impl<E, O, A> StepProcessor<E> for SimpleStepProcessor<E, O, A>
where
    E: Env,
    O: BatchBase + From<E::Obs>,
    A: BatchBase + From<E::Act>,
{
    type Config = SimpleStepProcessorConfig;
    type Output = GenericTransitionBatch<O, A>;

    /// Creates a new step processor with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `_config` - The configuration for the processor
    ///
    /// # Returns
    ///
    /// A new instance of the step processor
    fn build(_config: &Self::Config) -> Self {
        Self {
            prev_obs: None,
            phantom: PhantomData,
        }
    }

    /// Resets the processor with an initial observation.
    ///
    /// This method must be called before processing any steps to initialize
    /// the processor with the starting state of the environment.
    ///
    /// # Arguments
    ///
    /// * `init_obs` - The initial observation from the environment
    fn reset(&mut self, init_obs: E::Obs) {
        self.prev_obs = Some(init_obs.into());
    }

    /// Processes a step from the environment into a transition.
    ///
    /// This method converts an environment step into a transition suitable
    /// for training. It handles:
    /// - Converting observations and actions to the appropriate batch types
    /// - Managing the previous observation for constructing transitions
    /// - Handling episode termination and truncation
    ///
    /// # Arguments
    ///
    /// * `step` - The step to process
    ///
    /// # Returns
    ///
    /// A transition batch containing the processed step
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - The step contains more than one observation
    /// - `reset()` has not been called before processing steps
    /// - The step is terminal but does not contain an initial observation
    fn process(&mut self, step: crate::Step<E>) -> Self::Output {
        assert_eq!(step.obs.len(), 1);

        let batch = if self.prev_obs.is_none() {
            panic!("prev_obs is not set. Forgot to call reset()?");
        } else {
            let is_done = step.is_done();
            let next_obs = step.obs.clone().into();
            let obs = self.prev_obs.replace(step.obs.into()).unwrap();
            let act = step.act.into();
            let reward = step.reward;
            let is_terminated = step.is_terminated;
            let is_truncated = step.is_truncated;
            let ix_sample = None;
            let weight = None;

            if is_done {
                self.prev_obs
                    .replace(step.init_obs.expect("Failed to unwrap init_obs").into());
            }

            GenericTransitionBatch {
                obs,
                act,
                next_obs,
                reward,
                is_terminated,
                is_truncated,
                ix_sample,
                weight,
            }
        };

        batch
    }
}
