use std::cell::RefCell;
use std::fmt::Debug;

pub trait Obs: Clone {
    fn new() -> Self;
}

pub trait Info {}

pub trait Env {
    type Obs: Obs;
    type Act;
    type Info: Info;
    type ERR: Debug;

    fn step(&self, a: &Self::Act) -> (Self::Obs, f32, bool, Self::Info);

    fn reset(&self) -> Result<Self::Obs, Self::ERR>;
}

pub trait Policy<E: Env> {
    fn sample(&self, obs: &E::Obs) -> E::Act;
}

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

    pub fn sample(&self, n: usize) {
        let mut obs = match self.obs.replace(None) {
            None => self.env.reset().unwrap(),
            Some(obs) => obs.clone()
        };
        let mut done_last = false;

        for _ in 0..n {
            let a = self.pi.sample(&obs);
            let (o, _r, done, _info) = self.env.step(&a);
            obs = if done { self.env.reset().unwrap() } else { o };
            done_last = done;
        }

        if done_last {
            self.obs.replace(None);
        }
        else {
            self.obs.replace(Some(obs));
        }
    }
}
