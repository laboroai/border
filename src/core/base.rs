use std::{fmt::Debug, path::Path, error};

/// Represents an observation of the environment.
pub trait Obs: Clone {
    fn new() -> Self;
}

/// Represents an action of the environment.
pub trait Act: Clone {
}

/// Represents additional information to `Obs` and `Act`.
pub trait Info {}

/// Represents all information given at every step of agent-envieronment interaction.
pub struct Step<E: Env> {
    pub act: E::Act,
    pub obs: E::Obs,
    pub reward: f32,
    pub is_done: bool,
    pub info: E::Info,
}

impl<E: Env> Step<E> {
    pub fn new(obs: E::Obs, act: E::Act, reward: f32, is_done: bool, info: E::Info) -> Self {
        Step {
            act,
            obs,
            reward,
            is_done,
            info,
        }
    }
}

/// Represents an environment, typically an MDP.
pub trait Env {
    type Obs: Obs;
    type Act: Act;
    type Info: Info;
    type ERR: Debug;

    fn step(&self, a: &Self::Act) -> Step<Self> where Self: Sized;

    fn reset(&self) -> Result<Self::Obs, Self::ERR>;
}

/// Represents a policy. on an environment. It is based on a mapping from an observation
/// to an action. The mapping can be either of deterministic or stochastic.
pub trait Policy<E: Env> {
    /// Sample an action given an observation.
    fn sample(&self, obs: &E::Obs) -> E::Act;
}

/// Represents a trainable policy on an environment.
pub trait Agent<E: Env>: Policy<E> {
    /// Set the policy to training mode.
    fn train(&mut self);

    /// Set the policy to evaluation mode.
    fn eval(&mut self);

    /// Return if it is in training mode.
    fn is_train(&self) -> bool;

    /// Observe a [crate::core::base::Step] object.
    /// The agent is expected to do training its policy based on the observation.
    ///
    /// Return `true` if training of the agent is finished.
    /// TODO: Check the description. 
    fn observe(&mut self, step: Step<E>) -> bool;

    /// Push observation to the agent.
    /// This method is used when resetting the environment.
    fn push_obs(&self, obs: &E::Obs);

    /// Save the agent in the given directory.
    /// This method commonly creates a number of files consisting the agent
    /// into the given directory. For example, [crate::agents::dqn::DQN] agent saves
    /// two Q-networks corresponding to the original and target networks.
    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn error::Error>>;

    /// Load the agent from the given directory.
    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn error::Error>>;
}
