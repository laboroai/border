use super::Evaluator;
use crate::{Env, Policy};
use anyhow::Result;
use std::marker::PhantomData;

/// A default [`Evaluator`].
///
/// This evaluator runs a given number of episodes and returns the mean value of cumulative reward.
/// The code for this method is as follows:
///
/// ```no_run
/// # use anyhow::Result;
/// # use border_core::{Env, Policy, Evaluator};
/// # use std::marker::PhantomData;
/// # pub struct DefaultEvaluator<E: Env, P: Policy<E>> {
/// # n_episodes: usize,
/// # env: E,
/// # phantom: PhantomData<P>,
/// # }
///
/// # impl<E, P> DefaultEvaluator<E, P>
/// # where
/// # E: Env,
/// # P: Policy<E>,
/// # {
/// # fn run_episode(&mut self, policy: &mut P, init_obs: E::Obs) -> Result<f32> {
/// # unimplemented!();
/// # }
/// # }
/// # impl<E, P> Evaluator<E, P> for DefaultEvaluator<E, P>
/// # where
/// # E: Env,
/// # P: Policy<E>,
/// # {
/// # fn evaluate(&mut self, policy: &mut P) -> Result<f32> {
/// let mut r_total = 0f32;
///
/// for ix in 0..self.n_episodes {
///    let init_obs = self.env.reset_with_index(ix)?;
///    r_total += self.run_episode(policy, init_obs)?;
/// }
///
/// Ok(r_total / self.n_episodes as f32)
/// # }
/// # }
/// ```
pub struct DefaultEvaluator<E: Env, P: Policy<E>> {
    n_episodes: usize,
    env: E,
    phantom: PhantomData<P>,
}

impl<E, P> Evaluator<E, P> for DefaultEvaluator<E, P>
where
    E: Env,
    P: Policy<E>,
{
    fn evaluate(&mut self, policy: &mut P) -> Result<f32> {
        let mut r_total = 0f32;

        for ix in 0..self.n_episodes {
            let init_obs = self.env.reset_with_index(ix)?;
            r_total += self.run_episode(policy, init_obs)?;
        }

        Ok(r_total / self.n_episodes as f32)
    }
}

impl<E, P> DefaultEvaluator<E, P>
where
    E: Env,
    P: Policy<E>,
{
    /// Constructs [`DefaultEvaluator`].
    ///
    /// `env` - Instance of the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    ///   The evaluator returns the mean value of cumulative reward in each episode.
    pub fn new(env: E, n_episodes: usize) -> Result<Self> {
        Ok(Self {
            n_episodes,
            env,
            phantom: PhantomData,
        })
    }

    /// Runs an episode and returns the cumulative reward.
    fn run_episode(&mut self, policy: &mut P, init_obs: E::Obs) -> Result<f32> {
        let mut r_total = 0f32;
        let mut prev_obs = init_obs;

        loop {
            let act = policy.sample(&prev_obs);
            let (step, _) = self.env.step(&act);
            r_total += step.reward[0];
            if step.is_done() {
                break;
            }
            prev_obs = step.obs;
        }

        Ok(r_total)
    }
}
