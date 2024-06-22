//! A generic implementation of [`StepProcessor`](crate::StepProcessor).
use super::{StdBatch, BatchBase};
use crate::{Env, Obs, StepProcessor};
use std::{default::Default, marker::PhantomData};

/// Configuration of [`SimpleStepProcessor`].
#[derive(Clone, Debug)]
pub struct SimpleStepProcessorConfig {}

impl Default for SimpleStepProcessorConfig {
    fn default() -> Self {
        Self {}
    }
}

/// A generic implementation of [`StepProcessor`](crate::StepProcessor).
///
/// It supports 1-step TD backup for non-vectorized environment:
/// `E::Obs.len()` must be 1.
pub struct SimpleStepProcessor<E, O, A> {
    prev_obs: Option<O>,
    phantom: PhantomData<(E, A)>,
}

impl<E, O, A> StepProcessor<E> for SimpleStepProcessor<E, O, A>
where
    E: Env,
    O: BatchBase + From<E::Obs>,
    A: BatchBase + From<E::Act>,
{
    type Config = SimpleStepProcessorConfig;
    type Output = StdBatch<O, A>;

    fn build(_config: &Self::Config) -> Self {
        Self {
            prev_obs: None,
            phantom: PhantomData,
        }
    }

    fn reset(&mut self, init_obs: E::Obs) {
        self.prev_obs = Some(init_obs.into());
    }

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
                self.prev_obs.replace(step.init_obs.into());
            }

            StdBatch {
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
