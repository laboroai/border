use crate::{MinariConverter, MinariEnv};
use anyhow::Result;
use border_core::{Env, Evaluator, Policy};

/// An evaluator for the Point Maze environment.
///
/// It returns the mean value of cumulative rewards of all every episodes.
pub struct PointMazeEvaluator<T: MinariConverter> {
    n_episodes: usize,
    env: MinariEnv<T>,
}

impl<T: MinariConverter> Evaluator<MinariEnv<T>> for PointMazeEvaluator<T> {
    /// Returns the normalized score, i.e., normalized average return over episodes,
    /// if the environment has ref_min_score and ref_max_score.
    /// If not, returns the average return over episodes.
    fn evaluate<P: Policy<MinariEnv<T>>>(&mut self, policy: &mut P) -> Result<f32> {
        log::debug!("Evaluation");
        let mut r_total = 0f32;

        for ix in 0..self.n_episodes {
            log::trace!("Episode: {:?}", ix);
            let init_obs = self.env.reset_with_index(ix)?;
            let r = self.run_episode(policy, init_obs)?;
            log::trace!("Return : {:?}", r);
            r_total += r;
        }

        let score = r_total / self.n_episodes as f32;
        let score = match self.env.get_normalized_score(score) {
            Some(score) => Ok(score),
            None => Ok(score)
        };
        log::info!("Average return: {:?}", (r_total / self.n_episodes as f32));
        log::info!("Score         : {:?}", score);

        score
    }
}

impl<T: MinariConverter> PointMazeEvaluator<T> {
    /// Constructs [`PointMazeEvaluator`].
    ///
    /// `env` - Instance of the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    pub fn new(env: MinariEnv<T>, n_episodes: usize) -> Result<Self> {
        Ok(Self { n_episodes, env })
    }

    /// Runs an episode and returns the reward of the last step.
    fn run_episode<P: Policy<MinariEnv<T>>>(
        &mut self,
        policy: &mut P,
        init_obs: T::Obs,
    ) -> Result<f32> {
        let mut r_total = 0.0;
        let mut prev_obs = init_obs;
        let mut rewards = vec![];

        loop {
            let act = policy.sample(&prev_obs);
            let (step, _) = self.env.step(&act);
            r_total += step.reward[0];
            rewards.push(step.reward[0]);
            if step.is_done() {
                break;
            }
            prev_obs = step.obs;
        }

        log::trace!("Rewards: {:?}", rewards);
        log::debug!("Episode length: {:?}", rewards.len());
        log::debug!("Episode return: {:?}", r_total);

        Ok(r_total)
    }
}
