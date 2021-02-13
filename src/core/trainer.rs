use std::cell::RefCell;
use chrono::Local;
use log::info;

use crate::core::{
    Env, Agent,
    util::{sample, eval},
    record::{Recorder, RecordValue}
};

pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    env_eval: E,
    agent: A,
    obs_prev: RefCell<Option<E::Obs>>,
    max_opts: usize,
    eval_interval: usize,
    n_episodes_per_eval: usize,
    count_opts: usize,
    count_steps: usize
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    pub fn new(env: E, env_eval: E, agent: A) -> Self {
        Trainer {
            env,
            env_eval,
            agent,
            obs_prev: RefCell::new(None),
            max_opts: 0,
            eval_interval: 0,
            n_episodes_per_eval: 0,
            count_opts: 0,
            count_steps: 0,
        }
    }

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
        return (mean, min, max);
    }

    pub fn train<T: Recorder>(&mut self, recorder: &mut T) {
        let obs = self.env.reset(None).unwrap();
        self.agent.push_obs(&obs);
        self.obs_prev.replace(Some(obs));
        self.agent.train(); // set to training mode

        loop {
            // For resetted environments, elements in obs_prev are updated with env.reset().
            // See `sample()` in `util.rs`.
            let (step, _) = sample(&mut self.env, &mut self.agent, &self.obs_prev);
            self.count_steps += 1;

            // agent.observe() internally creates transisions, i.e., (o_t, a_t, o_t+1, r_t+1).
            let option_record = self.agent.observe(step);

            // For resetted environments, previous observation are updated.
            // This is required to make transisions consistend.
            self.agent.push_obs(&self.obs_prev.borrow().as_ref().unwrap());

            if let Some(mut record) = option_record {
                use RecordValue::{Scalar, DateTime};

                self.count_opts += 1;
                record.insert("n_steps", Scalar(self.count_steps as _));
                record.insert("n_opts", Scalar(self.count_opts as _));
                record.insert("datetime", DateTime(Local::now()));

                if self.count_opts % self.eval_interval == 0 {
                    self.agent.eval();

                    let rewards = eval(&mut self.env_eval, &mut self.agent, self.n_episodes_per_eval);
                    let (mean, min, max) = Self::stats_eval_reward(&rewards);
                    info!("Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
                        self.count_opts, mean, min, max);
                    record.insert("mean_cum_eval_reward", Scalar(mean as f64));
        
                    self.agent.train();
                }

                recorder.write(record);
            }

            if self.count_opts >= self.max_opts {
                break;
            }
        }
    }
}
