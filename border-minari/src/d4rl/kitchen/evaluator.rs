use anyhow::Result;
use border_core::{record::Record, Agent, Env, Evaluator, ReplayBufferBase};

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

impl<E> KitchenEvaluator<E>
where
    E: Env,
{
    /// Constructs [`KitchenEvaluator`].
    ///
    /// `env` - Instance of the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    pub fn new(config: &E::Config, n_episodes: usize) -> Result<Self> {
        Ok(Self {
            n_episodes,
            env: E::build(config, 42)?,
        })
    }
}
