use std::cell::RefCell;
use std::fmt::Debug;

pub trait Obs: Clone {
    fn new() -> Self;
}

pub trait Info {}

pub struct Step<O: Obs, I: Info> {
    obs: O,
    reward: f32,
    is_done: bool,
    info: I
}

impl<O: Obs, I: Info> Step<O, I> {
    pub fn new(obs: O, reward: f32, is_done: bool, info: I) -> Self {
        Step {
            obs,
            reward,
            is_done,
            info
        }
    }
}

pub trait Env {
    type Obs: Obs;
    type Act;
    type Info: Info;
    type ERR: Debug;

    // fn step(&self, a: &Self::Act) -> (Self::Obs, f32, bool, Self::Info);
    fn step(&self, a: &Self::Act) -> Step<Self::Obs, Self::Info>;

    fn reset(&self) -> Result<Self::Obs, Self::ERR>;
}

pub trait Policy<E: Env> {
    fn sample(&self, obs: &E::Obs) -> E::Act;
    fn train(&mut self);
    fn eval(&mut self);
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

    pub fn sample(&self, n: usize) -> Step<E::Obs, E::Info> {
        let mut obs = match self.obs.replace(None) {
            None => self.env.reset().unwrap(),
            Some(obs) => obs.clone()
        };

        let mut done_last;
        let mut step;
        let mut i = 0;

        loop {
            let a = self.pi.sample(&obs);
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

        Step::new(obs, step.reward, step.is_done, step.info)
    }
}

pub trait Agent<E: Env>: Policy<E> {
    /// Return `true` if training of the agent is finished.
    fn observe(&self, step: Step<E::Obs, E::Info>) -> bool;
}

pub struct Trainer<E: Env, A: Agent<E>> {
    sampler: Sampler<E, A>,
    agent: A
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    pub fn train(&self) {
        loop {
            let step = self.sampler.sample(1);
            let is_finished = self.agent.observe(step);
            if is_finished { break; }
        }
    }
}
