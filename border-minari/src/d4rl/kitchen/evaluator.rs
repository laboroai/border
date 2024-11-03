use anyhow::Result;
use border_core::{Env, Evaluator, Policy};

/// An evaluator for the Kitchen environment.
///
/// Basically, this evaluator works like the `DefaultEvaluator` from `border-core`.
/// However, it takes the reward value of the last step of each episode as the evaluation value,
/// instead of the cumulative reward used in the `DefaultEvaluator`.
/// This is because the reward of the Kitchen environment is the number of tasks
/// completed in the episode. Therefore, the cumulative reward is not a good metric for evaluation.
pub struct KitchenEvaluator<E: Env> {
    n_episodes: usize,
    env: E,
}

impl<E> Evaluator<E> for KitchenEvaluator<E>
where
    E: Env,
{
    fn evaluate<P: Policy<E>>(&mut self, policy: &mut P) -> Result<f32> {
        let mut r_total = 0f32;

        for ix in 0..self.n_episodes {
            let init_obs = self.env.reset_with_index(ix)?;
            r_total += self.run_episode(policy, init_obs)?;
        }

        Ok(r_total / self.n_episodes as f32)
    }
}

impl<E> KitchenEvaluator<E>
where
    E: Env,
{
    /// Constructs [`KitchenEvaluator`].
    ///
    /// `env` - Instance of the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    ///   The evaluator returns the mean value of cumulative reward in each episode.
    pub fn new(env: E, n_episodes: usize) -> Result<Self> {
        Ok(Self { n_episodes, env })
    }

    /// Runs an episode and returns the reward of the last step.
    fn run_episode<P: Policy<E>>(&mut self, policy: &mut P, init_obs: E::Obs) -> Result<f32> {
        let mut r_last;
        let mut prev_obs = init_obs;

        loop {
            let act = policy.sample(&prev_obs);
            let (step, _) = self.env.step(&act);
            r_last = step.reward[0];
            if step.is_done() {
                break;
            }
            prev_obs = step.obs;
        }

        Ok(r_last)
    }
}
