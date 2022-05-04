//! Replay buffer.
//use super::StdBatchBase;
use anyhow::Result;

/// Interface of buffers of experiences from environments.
///
/// Methods of this trait are used to push experiences.
pub trait ExperienceBufferBase {
    /// Items pushed into the buffer.
    type PushedItem;

    /// Pushes a transition into the buffer.
    fn push(&mut self, tr: Self::PushedItem) -> Result<()>;

    /// The number of samples in the buffer.
    fn len(&self) -> usize;
}

/// Interface of replay buffers.
///
/// This replay buffer generates a batch that 
pub trait ReplayBufferBase: ExperienceBufferBase {
    /// Configuration of the replay buffer.
    type Config: Clone;

    /// Batch generated from the buffer.
    type Batch;

    /// Build a replay buffer from [Self::Config].
    fn build(config: &Self::Config) -> Self;

    /// Constructs a batch.
    ///
    /// `beta` - The exponent for priority.
    fn batch(&mut self, size: usize) -> Result<Self::Batch>;

    /// Updates priority.
    ///
    /// Priority is commonly based on TD error.
    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_err: &Option<Vec<f32>>);
}
