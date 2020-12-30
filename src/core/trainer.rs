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

    /// The agent take an action and apply it to the environment.
    /// Then return [crate::core::base::Step] object.
    fn sample(&self) -> Step<E::Obs, E::Act, E::Info> {
        let obs = match self.obs.replace(None) {
            None => {
                let obs = self.env.reset().unwrap();
                self.agent.push_obs(&obs);
                obs
            },
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
