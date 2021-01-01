use std::cell::RefCell;
use log::{info};
use crate::core::{Env, Agent, util::{sample, eval}};

pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    env_eval: E,
    agent: A,
    obs: RefCell<Option<E::Obs>>,
    max_opts: usize,
    n_opts_per_eval: usize,
    n_episodes_per_eval: usize,
    count_opts: usize,
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    pub fn new(env: E, env_eval: E, agent: A) -> Self {
        Trainer {
            env,
            env_eval,
            agent,
            obs: RefCell::new(None),
            max_opts: 0,
            n_opts_per_eval: 0,
            n_episodes_per_eval: 0,
            count_opts: 0,
        }
    }

    pub fn max_opts(mut self, v: usize) -> Self {
        self.max_opts = v;
        self
    }

    pub fn n_opts_per_eval(mut self, v: usize) -> Self {
        self.n_opts_per_eval = v;
        self
    }

    pub fn n_episodes_per_eval(mut self, v: usize) -> Self {
        self.n_episodes_per_eval = v;
        self
    }

    pub fn get_agent(&self) -> &impl Agent<E> {
        &self.agent
    }

    pub fn eval(&mut self) {
        // TODO: check the maximum number of steps of the environment for evaluation.
        // If it is infinite, the number of evaluation steps should be given in place of
        // n_episodes_per_eval.
        self.agent.eval();
        let mut rs = Vec::new();

        for _ in 0..self.n_episodes_per_eval {
            let mut r_sum = 0.0;
            let obs = self.env_eval.reset().unwrap();
            self.obs.replace(Some(obs));    
            loop {
                let step = sample(&self.env_eval, &mut self.agent, &self.obs);
                r_sum += step.reward;
                if step.is_done { break; }
            }
            rs.push(r_sum);
        }

        let mean: f32 = rs.iter().sum::<f32>() / self.n_episodes_per_eval as f32;
        let min = rs.iter().fold(f32::NAN, |m, v| v.min(m));
        let max = rs.iter().fold(f32::NAN, |m, v| v.max(m));
        info!("Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
            self.count_opts, mean, min, max);
    }

    pub fn train(&mut self) {
        self.agent.train(); // set to training mode
        loop {
            let step = sample(&self.env, &mut self.agent, &self.obs);
            let is_optimized = self.agent.observe(step);
            if is_optimized {
                self.count_opts += 1;
                if self.count_opts % self.n_opts_per_eval == 0 {
                    // self.eval();
                    eval(&self.env_eval, &mut self.agent, self.n_episodes_per_eval, Some(self.count_opts));
                    self.agent.train();
                }
            }
            if self.count_opts >= self.max_opts {
                break;
            }
        }
    }
}
