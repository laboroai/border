use super::Evaluator;
use crate::{Agent, Env, ReplayBufferBase};
use anyhow::Result;

/// A default [`Evaluator`].
///
/// This struct runs episodes of the given number of times.
pub struct DefaultEvaluator<E: Env> {
    n_episodes: usize,
    env: E,
}

impl<E: Env> Evaluator<E> for DefaultEvaluator<E> {
    fn evaluate<R>(&mut self, policy: &mut Box<dyn Agent<E, R>>) -> Result<f32>
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

        Ok(r_total / self.n_episodes as f32)
    }
}

impl<E: Env> DefaultEvaluator<E>
where
    E: Env,
{
    /// Constructs [`DefaultEvaluator`].
    ///
    /// `config` - Configuration of the environment.
    /// `seed` - Random seed, which will be used to create the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    ///   The evaluator returns the mean value of cumulative reward in each episode.
    pub fn new(config: &E::Config, seed: i64, n_episodes: usize) -> Result<Self> {
        Ok(Self {
            n_episodes,
            env: E::build(config, seed)?,
        })
    }
}
