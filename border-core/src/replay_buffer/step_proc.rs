//! A generic implementation of [StepProcessorBase](crate::StepProcessorBase).
use std::{default::Default, marker::PhantomData};
use crate::{Obs, Env, StepProcessorBase};
use super::{Batch, SubBatch};

/// Configuration of [SimpleStepProcessor].
#[derive(Clone, Debug)]
pub struct SimpleStepProcessorConfig {
    // gamma: f32,
}

impl Default for SimpleStepProcessorConfig {
    fn default() -> Self {
        Self {
            // gamma: 0.99
        }
    }
}

/// A generic implementation of [StepProcessorBase](crate::StepProcessorBase).
///
/// It supports 1-step TD backup for non-vectorized environment:
/// `E::Obs.len()` must be 1.
pub struct SimpleStepProcessor<E, O, A> {
    // gamma: f32,
    prev_obs: Option<O>,
    phantom: PhantomData<(E, A)>
}

impl<E, O, A> StepProcessorBase<E> for SimpleStepProcessor<E, O, A>
where
    E: Env,
    O: SubBatch + From<E::Obs>,
    A: SubBatch + From<E::Act>,
{
    type Config = SimpleStepProcessorConfig;
    type Output = Batch<O, A>;

    fn build(_config: &Self::Config) -> Self {
        Self {
            // gamma: config.gamma,
            prev_obs: None,
            phantom: PhantomData
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
            let next_obs = step.obs.clone().into();
            let obs = self.prev_obs.replace(step.obs.into()).unwrap();
            let act = step.act.into();
            let reward = step.reward;
            let is_done = step.is_done;
            let ix_sample = None;
            let weight = None;

            if is_done[0] == 1 {
                self.prev_obs.replace(step.init_obs.into());
            }

            Batch {obs, act, next_obs, reward, is_done, ix_sample, weight}
        };

        batch
    }
}
