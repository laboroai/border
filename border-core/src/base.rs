//! Core functionalities.
mod agent;
mod batch;
mod env;
mod policy;
mod step;
mod replay_buffer;
pub use agent::Agent;
pub use batch::Batch;
pub use env::Env;
pub use policy::Policy;
use std::fmt::Debug;
pub use step::{Info, Step, StepProcessorBase};
pub use replay_buffer::ReplayBufferBase;

/// Represents a set of observations of the environment.
pub trait Obs: Clone + Debug {
    /// Returns a dummy observation.
    ///
    /// The observation created with this method is ignored.
    fn dummy(n: usize) -> Self;

    /// Replace elements of observation where `is_done[i] == 1.0`.
    /// This method assumes that `is_done.len() == n_procs`.
    fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self;

    /// Returns the number of observations in the object.
    fn len(&self) -> usize;
}

/// Represents a set of actions of the environment.
pub trait Act: Clone + Debug {
    /// Returns the number of actions in the object.
    fn len(&self) -> usize;
}
