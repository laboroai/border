use std::fmt::Debug;

pub trait Obs: Clone {
    fn new() -> Self;
}

pub trait Info {}

pub struct Step<O: Obs, I: Info> {
    pub(in crate::core) obs: O,
    pub(in crate::core) reward: f32,
    pub(in crate::core) is_done: bool,
    pub(in crate::core) info: I
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

    fn step(&self, a: &Self::Act) -> Step<Self::Obs, Self::Info>;

    fn reset(&self) -> Result<Self::Obs, Self::ERR>;
}

pub trait Policy<E: Env> {
    fn sample(&self, obs: &E::Obs) -> E::Act;
    fn train(&mut self);
    fn eval(&mut self);
}

pub trait Agent<E: Env>: Policy<E> {
    /// Observe a [crate::core::base::Step] object.
    /// The agent is expected to do training its policy based on the observation.
    ///
    /// Return `true` if training of the agent is finished.
    /// TODO: Check the description. 
    fn observe(&self, step: Step<E::Obs, E::Info>) -> bool;

    /// Push observation to the agent.
    /// This method is used when resetting the environment.
    fn push_obs(&self, obs: &E::Obs);
}
