use std::cell::RefCell;
use crate::core::{Env, Agent, Step};

pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    agent: A,
    obs: RefCell<Option<E::Obs>>,
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    pub fn new(env: E, agent: A) -> Self {
        Trainer {
            env,
            agent,
            obs: RefCell::new(None)
        }
    }

    fn sample(&self) -> Step<E::Obs, E::Info> {
        let obs = match self.obs.replace(None) {
            None => self.env.reset().unwrap(),
            Some(obs) => obs.clone()
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
        // Step::new(obs, step.reward, step.is_done, step.info)
    }

    pub fn train(&mut self) {
        self.agent.train();
        loop {
            let step = self.sample();
            let is_finished = self.agent.observe(step);
            if is_finished { break; }
        }
    }
}
