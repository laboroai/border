use anyhow::Result;
use border_core::{record::Record, Agent, Env, Evaluator, ReplayBufferBase};

/// An evaluator for the AntMaze environment.
///
/// It returns the mean value of cumulative rewards of all every episodes.
pub struct AntMazeEvaluator<E: Env> {
    n_episodes: usize,
    env: E,
}

impl<E> Evaluator<E> for AntMazeEvaluator<E>
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

        let name = format!("Average return over {} episodes", self.n_episodes);
        Ok(Record::from_scalar(name, r_total / self.n_episodes as f32))
    }
}

impl<E: Env> AntMazeEvaluator<E> {
    /// Constructs [`AntMazeEvaluator`].
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
