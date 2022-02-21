use serde::{Deserialize, Serialize};

/// Configuration of [ActorManager](super::ActorManager).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActorManagerConfig {
    /// Number of samples to be buffered in each actor until being pushed to the replay buffer.
    ///
    /// The default value is 100.
    pub n_buffer: usize,
}

impl Default for ActorManagerConfig {
    fn default()  -> Self {
        Self {
            n_buffer: 100,
        }
    }
}
