use std::cell::RefCell;
use crate::core::{Env, Policy, Step};

pub struct Sampler<E: Env, P: Policy<E>> {
    env: E,
    pi : P,
    obs: RefCell<Option<E::Obs>>,
}

impl<E: Env, P: Policy<E>> Sampler<E, P> {
    pub fn new(env: E, pi: P) -> Self {
        Sampler {
            env,
            pi,
            obs: RefCell::new(None)
        }
    }

    pub fn sample(&self, n: usize) -> Step<E::Obs, E::Act, E::Info> {
        let mut obs = match self.obs.replace(None) {
            None => self.env.reset().unwrap(),
            Some(obs) => obs.clone()
        };

        let mut done_last;
        let mut step;
        let mut i = 0;
        let mut a;

        loop {
            a = self.pi.sample(&obs);
            step = self.env.step(&a);
            obs = if step.is_done { self.env.reset().unwrap() } else { step.obs };
            done_last = step.is_done;
            if i == n - 1 {
                break;
            }
            else {
                i += 1;
            }
        }

        if done_last {
            self.obs.replace(None);
        }
        else {
            self.obs.replace(Some(obs.clone()));
        }

        Step::new(obs, a, step.reward, step.is_done, step.info)
    }
}
