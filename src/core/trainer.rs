use std::cell::RefCell;
use chrono::Local;
use log::info;

use crate::core::{
    Env, Agent,
    util::{sample, eval},
    record::{Recorder, RecordValue}
};

pub struct TrainerBuilder {
    max_opts: usize,
    eval_interval: usize,
    n_episodes_per_eval: usize,
    eval_threshold: Option<f32>,
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        Self {
            max_opts: 0,
            eval_interval: 0,
            n_episodes_per_eval: 0,
            eval_threshold: None,
        }
    }    
}

impl TrainerBuilder {
    pub fn max_opts(mut self, v: usize) -> Self {
        self.max_opts = v;
        self
    }

    pub fn eval_interval(mut self, v: usize) -> Self {
        self.eval_interval = v;
        self
    }

    pub fn n_episodes_per_eval(mut self, v: usize) -> Self {
        self.n_episodes_per_eval = v;
        self
    }

    pub fn eval_threshold(mut self, v: f32) -> Self {
        self.eval_threshold = Some(v);
        self
    }

    pub fn build<E, A>(self, env: E, env_eval: E, agent: A) -> Trainer<E, A> where
        E: Env,
        A: Agent<E>
    {
        Trainer {
            env,
            env_eval,
            agent,
            obs_prev: RefCell::new(None),
            max_opts: self.max_opts,
            eval_interval: self.eval_interval,
            n_episodes_per_eval: self.n_episodes_per_eval,
            eval_threshold: self.eval_threshold,
            count_opts: 0,
            count_steps: 0,    
        }
    }
}

pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    env_eval: E,
    agent: A,
    obs_prev: RefCell<Option<E::Obs>>,
    max_opts: usize,
    eval_interval: usize,
    n_episodes_per_eval: usize,
    eval_threshold: Option<f32>,
    count_opts: usize,
    count_steps: usize
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    pub fn get_agent(&self) -> &impl Agent<E> {
        &self.agent
    }

    pub fn get_env(&self) -> &E {
        &self.env
    }

    pub fn get_env_eval(&self) -> &E {
        &self.env_eval
    }

    fn stats_eval_reward(rs: &Vec<f32>) -> (f32, f32, f32) {
        let mean: f32 = rs.iter().sum::<f32>() / (rs.len() as f32);
        let min = rs.iter().fold(f32::NAN, |m, v| v.min(m));
        let max = rs.iter().fold(f32::NAN, |m, v| v.max(m));

        (mean, min, max)
    }

    /// Train the agent.
    ///
    /// In the training loop, the following values are recorded in the `recorder`:
    /// * `n_steps` - The nunber of steps interacting with the environment.
    /// * `n_opts` - The number of optimization steps.
    /// * `datetime` - `Date and time`.
    /// * `mean_cum_eval_reward` - Cumulative rewards in evaluation runs.
    pub fn train<T: Recorder>(&mut self, recorder: &mut T) {
        let mut count_steps_local = 0;
        let mut now = std::time::SystemTime::now();

        let obs = self.env.reset(None).unwrap();
        self.agent.push_obs(&obs);
        self.obs_prev.replace(Some(obs));
        self.agent.train(); // set to training mode

        loop {
            let mut over_eval_threshold = false;

            // For resetted environments, elements in obs_prev are updated with env.reset().
            // After the update, obs_prev will have o_t+1 without reset, or o_0 with reset.
            // See `sample()` in `util.rs`.
            let (step, _) = sample(&mut self.env, &mut self.agent, &self.obs_prev);
            self.count_steps += 1;
            count_steps_local += 1;

            // agent.observe() internally creates transisions, i.e., (o_t, a_t, o_t+1, r_t+1).
            // For o_t, the previous observation stored in the agent is used.
            let option_record = self.agent.observe(step);

            // The previous observations in the agent are updated with obs_prev.
            // These are o_t+1 (without reset) or o_0 (with reset).
            // In the next iteration of the loop, o_t+1 will be treated as the previous observation
            // in the next training step executed in agent.observation().
            self.agent.push_obs(&self.obs_prev.borrow().as_ref().unwrap());

            if let Some(mut record) = option_record {
                use RecordValue::{Scalar, DateTime};

                self.count_opts += 1;
                record.insert("n_steps", Scalar(self.count_steps as _));
                record.insert("n_opts", Scalar(self.count_opts as _));
                record.insert("datetime", DateTime(Local::now()));

                if self.count_opts % self.eval_interval == 0 {
                    // Show FPS before evaluation
                    let fps = match now.elapsed() {
                        Ok(elapsed) => {
                            Some(count_steps_local as f32 / elapsed.as_millis() as f32 * 1000.0)
                        },
                        Err(_) => {
                            None
                        }
                    };
                    // Reset counter for getting FPS in training
                    count_steps_local = 0;

                    // The timer is used to measure the elapsed time for evaluation
                    now = std::time::SystemTime::now();

                    // Evaluation
                    self.agent.eval();
                    let rewards = eval(&mut self.env_eval, &mut self.agent, self.n_episodes_per_eval);
                    let (mean, min, max) = Self::stats_eval_reward(&rewards);
                    info!("Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
                        self.count_opts, mean, min, max);
                    record.insert("mean_cum_eval_reward", Scalar(mean));

                    if let Some(fps) = fps {
                        info!("{} FPS in training", fps);
                    }

                    match now.elapsed() {
                        Ok(elapsed) => {
                            info!("{} sec. in evaluation", elapsed.as_millis() as f32 / 1000.0);
                        },
                        Err(_) => {
                            info!("An error occured when getting time")
                        }
                    }

                    // The timer is used to measure the elapsed time for training
                    now = std::time::SystemTime::now();

                    self.agent.train();

                    if let Some(th) = self.eval_threshold {
                        over_eval_threshold = mean >= th;
                    }
                }

                recorder.write(record);
            }

            if self.count_opts >= self.max_opts || over_eval_threshold {
                break;
            }
        }
    }
}
