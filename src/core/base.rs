use std::{fmt::Debug, path::Path, error};

pub trait Obs: Clone {
    fn new() -> Self;
}

pub trait Act: Clone {
}

pub trait Info {}

pub struct Step<O: Obs, A: Act, I: Info> {
    pub act: A,
    pub obs: O,
    pub reward: f32,
    pub is_done: bool,
    pub info: I,
}

impl<O: Obs, A: Act, I: Info> Step<O, A, I> {
    pub fn new(obs: O, act: A, reward: f32, is_done: bool, info: I) -> Self {
        Step {
            act,
            obs,
            reward,
            is_done,
            info,
        }
    }
}

pub trait Env {
    type Obs: Obs;
    type Act: Act;
    type Info: Info;
    type ERR: Debug;

    fn step(&self, a: &Self::Act) -> Step<Self::Obs, Self::Act, Self::Info>;

    fn reset(&self) -> Result<Self::Obs, Self::ERR>;
}

pub trait Policy<E: Env> {
    fn sample(&self, obs: &E::Obs) -> E::Act;
    fn train(&mut self);
    fn eval(&mut self);
    fn is_train(&self) -> bool;
}

pub trait Agent<E: Env>: Policy<E> {
    /// Observe a [crate::core::base::Step] object.
    /// The agent is expected to do training its policy based on the observation.
    ///
    /// Return `true` if training of the agent is finished.
    /// TODO: Check the description. 
    fn observe(&mut self, step: Step<E::Obs, E::Act, E::Info>) -> bool;

    /// Push observation to the agent.
    /// This method is used when resetting the environment.
    fn push_obs(&self, obs: &E::Obs);

    /// Save the model in the given directory
    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn error::Error>>;

    // /// Load the model from the given directory
    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn error::Error>>;
}
