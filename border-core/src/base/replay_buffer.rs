//! Replay buffer.
use super::Batch;
use anyhow::Result;

/// Represents a replay buffer.
pub trait ReplayBufferBase {
    /// Configuration of the replay buffer.
    type Config: Clone;

    /// Items pushed into the buffer.
    type PushedItem;

    /// Batch generated from the buffer.
    type Batch: Batch;

    /// Build a replay buffer from [Self::Config].
    fn build(config: &Self::Config) -> Self;

    /// The number of samples in the buffer.
    fn len(&self) -> usize;

    /// Constructs a batch.
    ///
    /// `beta` - The exponent for priority.
    fn batch(&self, size: usize) -> Result<Self::Batch>;

    /// Pushes a transition into the buffer.
    fn push(&mut self, tr: Self::PushedItem);

    /// Updates priority based on TD error.
    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_err: &Option<Vec<f32>>);
}
