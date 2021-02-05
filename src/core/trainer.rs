use std::cell::RefCell;
use crate::core::{Env, Agent, util::{sample, eval}};

pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    env_eval: E,
    agent: A,
    obs_prev: RefCell<Option<E::Obs>>,
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
            obs_prev: RefCell::new(None),
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

    pub fn get_env(&self) -> &E {
        &self.env
    }

    pub fn get_env_eval(&self) -> &E {
        &self.env_eval
    }

    pub fn train(&mut self) {
        let obs = self.env.reset(None).unwrap();
        self.agent.push_obs(&obs);
        self.obs_prev.replace(Some(obs));
        self.agent.train(); // set to training mode

        loop {
            // For resetted environments, elements in obs_prev are updated with env.reset().
            // See `sample()` in `util.rs`.
            let step = sample(&mut self.env, &mut self.agent, &self.obs_prev);
            // agent.observe() internally creates transisions, i.e., (o_t, a_t, o_t+1, r_t+1).
            let is_optimized = self.agent.observe(step);
            // For resetted environments, previous observation are updated.
            // This is required to make transisions consistend.
            self.agent.push_obs(&self.obs_prev.borrow().as_ref().unwrap());

            if is_optimized {
                self.count_opts += 1;
                if self.count_opts % self.n_opts_per_eval == 0 {
                    self.agent.eval();
                    eval(&mut self.env_eval, &mut self.agent, self.n_episodes_per_eval, Some(self.count_opts));
                    self.agent.train();
                }
            }
            if self.count_opts >= self.max_opts {
                break;
            }
        }
    }
}
