//! Configuration of [SimpleReplayBuffer](super::SimpleReplayBuffer).
use std::default::Default;

/// Configuration of [SimpleReplayBuffer](super::SimpleReplayBuffer).
pub struct SimpleReplayBufferConfig {
    pub(super) capacity: usize,
    pub(super) seed: u64,
}

impl Default for SimpleReplayBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            seed: 42,
        }
    }
}

impl SimpleReplayBufferConfig {
    /// Sets the capacity of the replay buffer.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
}
