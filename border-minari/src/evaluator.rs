//! Evaluator for Minari environments.
use crate::{MinariConverter, MinariEnv};
use anyhow::Result;
use border_core::{record::Record, Agent, Env, Evaluator, ReplayBufferBase};

/// An evaluator for Minari environments.
///
/// This struct implements the `Evaluator` trait for Minari environments.
/// This struct evaluates the policy on the Minari environment for a given number of episodes.
/// The average return over episodes is returned.
/// If the environment has ref_min_score and ref_max_score, the normalized score is also returned
/// in the record.
pub struct MinariEvaluator<T: MinariConverter> {
    n_episodes: usize,
    env: MinariEnv<T>,
}

impl<T: MinariConverter> Evaluator<MinariEnv<T>> for MinariEvaluator<T> {
    /// Evaluates the policy on the Minari environment.
    ///
    /// This function evaluates the policy on the Minari environment for a given number of episodes.
    /// The average return over episodes is returned.
    /// If the environment has ref_min_score and ref_max_score, the normalized score is also returned
    /// in the record.
    fn evaluate<R: ReplayBufferBase>(
        &mut self,
        policy: &mut Box<dyn Agent<MinariEnv<T>, R>>,
    ) -> Result<(f32, Record)> {
        log::debug!("Evaluation");
        let mut r_total = 0f32;

        // Episode loop
        for ix in 0..self.n_episodes {
            log::trace!("Episode: {:?}", ix);
            let mut prev_obs = self.env.reset_with_index(ix)?;

            // Environment loop
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

        // Average return
        let name = format!("Episode return");
        let score = r_total / self.n_episodes as f32;
        let mut record = Record::from_scalar(name, score);

        // Normalized score
        if let Some(score) = self.env.get_normalized_score(score) {
            // record = record.merge(Record::from_scalar("Nomalized score", score));
            let name = "Normalized score";
            record = Record::from_scalar(name, score);
        }

        Ok((score, record))
    }
}

impl<T: MinariConverter> MinariEvaluator<T> {
    /// Constructs [`MinariEvaluator`].
    ///
    /// `env` - Instance of the environment.
    /// `n_episodes` - The number of episodes for evaluation.
    pub fn new(env: MinariEnv<T>, n_episodes: usize) -> Result<Self> {
        Ok(Self { n_episodes, env })
    }
}
