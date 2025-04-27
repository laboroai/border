//! Replay buffer interface for reinforcement learning.
//!
//! This module defines the core interfaces for experience replay buffers in reinforcement learning.
//! Replay buffers are essential components that store and sample experiences (transitions)
//! for training agents, enabling more efficient learning through experience replay.

use anyhow::Result;

/// Interface for buffers that store experiences from environments.
///
/// This trait defines the basic operations for storing experiences in a buffer.
/// It is typically used by processes that need to sample experiences for training.
///
/// # Type Parameters
///
/// * `Item` - The type of experience stored in the buffer
///
/// # Examples
///
/// ```ignore
/// struct SimpleBuffer<T> {
///     items: Vec<T>,
/// }
///
/// impl<T> ExperienceBufferBase for SimpleBuffer<T> {
///     type Item = T;
///
///     fn push(&mut self, tr: T) -> Result<()> {
///         self.items.push(tr);
///         Ok(())
///     }
///
///     fn len(&self) -> usize {
///         self.items.len()
///     }
/// }
/// ```
pub trait ExperienceBufferBase {
    /// The type of items stored in the buffer.
    ///
    /// This can be any type that represents an experience or transition
    /// from the environment.
    type Item;

    /// Pushes a new experience into the buffer.
    ///
    /// # Arguments
    ///
    /// * `tr` - The experience to store
    ///
    /// # Returns
    ///
    /// `Ok(())` if the push was successful, or an error if it failed
    fn push(&mut self, tr: Self::Item) -> Result<()>;

    /// Returns the current number of experiences in the buffer.
    ///
    /// # Returns
    ///
    /// The number of experiences currently stored
    fn len(&self) -> usize;
}

/// Interface for replay buffers that generate batches for training.
///
/// This trait provides functionality for sampling batches of experiences
/// for training agents. It is independent of [`ExperienceBufferBase`] and
/// focuses solely on the batch generation process.
///
/// # Associated Types
///
/// * `Config` - Configuration parameters for the buffer
/// * `Batch` - The type of batch generated for training
pub trait ReplayBufferBase {
    /// Configuration parameters for the replay buffer.
    ///
    /// This type must implement `Clone` to support building multiple instances
    /// with the same configuration.
    type Config: Clone;

    /// The type of batch generated for training.
    ///
    /// This is typically a collection of experiences that can be used
    /// directly for training an agent.
    type Batch;

    /// Builds a new replay buffer from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration parameters
    ///
    /// # Returns
    ///
    /// A new instance of the replay buffer
    fn build(config: &Self::Config) -> Self;

    /// Constructs a batch of experiences for training.
    ///
    /// This method samples experiences from the buffer and returns them
    /// in a format suitable for training.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of experiences to include in the batch
    ///
    /// # Returns
    ///
    /// A batch of experiences or an error if sampling failed
    fn batch(&mut self, size: usize) -> Result<Self::Batch>;

    /// Updates the priorities of experiences in the buffer.
    ///
    /// This method is used in prioritized experience replay to adjust
    /// the sampling probabilities of experiences based on their TD errors.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Optional indices of experiences to update
    /// * `td_err` - Optional TD errors for the experiences
    ///
    /// # Note
    ///
    /// This method is optional and may be moved to a separate trait
    /// in future versions to better support non-prioritized replay buffers.
    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_err: &Option<Vec<f32>>);
}

/// A dummy replay buffer that does nothing.
///
/// This struct is used as a placeholder when a replay buffer is not needed.
pub struct NullReplayBuffer;

impl ReplayBufferBase for NullReplayBuffer {
    type Batch = ();
    type Config = ();

    #[allow(unused_variables)]
    fn build(config: &Self::Config) -> Self {
        Self
    }

    #[allow(unused_variables)]
    fn batch(&mut self, size: usize) -> Result<Self::Batch> {
        unimplemented!();
    }

    #[allow(unused_variables)]
    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_err: &Option<Vec<f32>>) {
        unimplemented!();
    }
}
