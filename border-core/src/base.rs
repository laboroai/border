//! Core functionalities.
mod agent;
mod batch;
mod env;
mod policy;
mod replay_buffer;
mod step;
pub use agent::Agent;
pub use batch::StdBatchBase;
pub use env::Env;
pub use policy::Policy;
pub use replay_buffer::{ExperienceBufferBase, ReplayBufferBase};
use std::fmt::Debug;
pub use step::{Info, Step, StepProcessorBase};

/// A set of observations of an environment.
///
/// Old versions of the library support vectorized environment and
/// [Obs] was able to handle multiple observations.
/// In the current version, no vectorized environment is implemented.
/// Thus, [Obs]`::len()` always returns 1.
pub trait Obs: Clone + Debug {
    /// Returns a dummy observation.
    ///
    /// The observation created with this method is ignored.
    fn dummy(n: usize) -> Self;

    /// Replace elements of observation where `is_done[i] == 1`.
    ///
    /// This method assumes that `is_done.len() == n_procs`.
    fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self;

    /// Returns the number of observations in the object.
    fn len(&self) -> usize;
}

/// A set of actions of the environment.
pub trait Act: Clone + Debug {
    /// Returns the number of actions in the object.
    fn len(&self) -> usize;
}
