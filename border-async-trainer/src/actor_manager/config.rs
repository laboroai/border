use serde::{Deserialize, Serialize};

/// Configuration of [ActorManager](super::ActorManager).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActorManagerConfig {
    /// The number of [Actor](crate::Actor)s.
    pub n_actors: usize,

    /// Number of samples to be buffered in each actor until being pushed to the replay buffer.
    ///
    /// The default value is 100.
    pub samples_per_push: usize,
}

impl ActorManagerConfig {
    pub fn new(n_actors: usize)  -> Self {
        Self {
            n_actors,
            samples_per_push: 100,
        }
    }
}
