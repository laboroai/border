use anyhow::Result;
use border_core::{Env, Evaluator, Policy};

/// An evaluator for the Point Maze environment.
///
/// It returns the mean value of cumulative rewards of all every episodes.
pub struct PointMazeEvaluator<E: Env> {
    n_episodes: usize,
    env: E,
}

impl<E> Evaluator<E> for PointMazeEvaluator<E>
where
    E: Env,
{
    /// Returns the mean value of cumulative rewards of all every episodes.
    fn evaluate<P: Policy<E>>(&mut self, policy: &mut P) -> Result<f32> {
        let mut r_total = 0f32;

        for ix in 0..self.n_episodes {
            let init_obs = self.env.reset_with_index(ix)?;
            r_total += self.run_episode(policy, init_obs)?;
        }

        Ok(r_total / self.n_episodes as f32)
    }
}

impl<E> PointMazeEvaluator<E>
where
    E: Env,
{
    /// Constructs [`PointMazeEvaluator`].
    ///
    /// `env` - Instance of the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    pub fn new(env: E, n_episodes: usize) -> Result<Self> {
        Ok(Self { n_episodes, env })
    }

    /// Runs an episode and returns the reward of the last step.
    fn run_episode<P: Policy<E>>(&mut self, policy: &mut P, init_obs: E::Obs) -> Result<f32> {
        let mut r_total = 0.0;
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
