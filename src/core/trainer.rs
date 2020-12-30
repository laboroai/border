use std::cell::RefCell;
use crate::core::{Env, Agent, Step};

pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    env_eval: E,
    agent: A,
    obs: RefCell<Option<E::Obs>>,
    max_opts: usize,
    count_opts: usize,
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    pub fn new(env: E, env_eval: E, agent: A, max_opts: usize) -> Self {
        Trainer {
            env,
            env_eval,
            agent,
            obs: RefCell::new(None),
            max_opts,
            count_opts: 0,
        }
    }

    /// The agent take an action and apply it to the environment.
    /// Then return [crate::core::base::Step] object.
    fn sample(&self) -> Step<E::Obs, E::Act, E::Info> {
        let obs = match self.obs.replace(None) {
            None => {
                let obs = self.env.reset().unwrap();
                self.agent.push_obs(&obs);
                obs
            },
            Some(obs) => obs
        };
        let a = self.agent.sample(&obs);
        let step = self.env.step(&a);

        if step.is_done {
            self.obs.replace(None);
        }
        else {
            self.obs.replace(Some(step.obs.clone()));
        }

        step
    }

    pub fn train(&mut self) {
        self.agent.train(); // set to training mode
        loop {
            let step = self.sample();
            let is_optimized = self.agent.observe(step);
            if is_optimized { self.count_opts += 1; }
            if self.count_opts >= self.max_opts { break; }
        }
    }
}
