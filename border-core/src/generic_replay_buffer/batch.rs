//! Generic implementation of transition batches for reinforcement learning.
//!
//! This module provides a generic implementation of transition batches that can handle
//! arbitrary observation and action types. It supports the following features:
//! - Efficient batch processing
//! - Weighting for prioritized experience replay
//! - Transition sampling and management

use crate::TransitionBatch;

/// A trait defining basic batch operations.
///
/// This trait provides fundamental operations for efficiently managing batches of
/// observations and actions.
///
/// # Type Parameters
///
/// * `Self` - The batch type, representing batches of observations or actions.
///
/// # Examples
///
/// ```ignore
/// struct TensorBatch {
///     data: Vec<f32>,
///     shape: Vec<usize>,
/// }
///
/// impl BatchBase for TensorBatch {
///     fn new(capacity: usize) -> Self {
///         Self {
///             data: Vec::with_capacity(capacity),
///             shape: vec![],
///         }
///     }
///
///     fn push(&mut self, ix: usize, data: Self) {
///         // Data addition logic
///     }
///
///     fn sample(&self, ixs: &Vec<usize>) -> Self {
///         // Sampling logic
///     }
/// }
/// ```
pub trait BatchBase {
    /// Creates a new batch with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity of the batch
    fn new(capacity: usize) -> Self;

    /// Adds data at the specified index.
    ///
    /// # Arguments
    ///
    /// * `ix` - Index where data should be added
    /// * `data` - Data to be added
    fn push(&mut self, ix: usize, data: Self);

    /// Retrieves samples from the specified indices.
    ///
    /// # Arguments
    ///
    /// * `ixs` - List of indices to sample from
    ///
    /// # Returns
    ///
    /// A new batch containing the sampled data
    fn sample(&self, ixs: &Vec<usize>) -> Self;
}

/// A generic structure representing transitions in reinforcement learning.
///
/// This structure efficiently manages reinforcement learning transitions
/// (observations, actions, rewards, etc.). It also includes support for
/// prioritized experience replay (PER).
///
/// # Type Parameters
///
/// * `O` - Observation type, must implement `BatchBase`
/// * `A` - Action type, must implement `BatchBase`
///
/// # Examples
///
/// ```ignore
/// let batch = GenericTransitionBatch::<Tensor, Tensor>::with_capacity(32);
/// ```
pub struct GenericTransitionBatch<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    /// Current observations
    pub obs: O,

    /// Selected actions
    pub act: A,

    /// Next state observations
    pub next_obs: O,

    /// Transition rewards
    pub reward: Vec<f32>,

    /// Episode termination flags
    pub is_terminated: Vec<i8>,

    /// Episode truncation flags
    pub is_truncated: Vec<i8>,

    /// Weights for prioritized experience replay
    pub weight: Option<Vec<f32>>,

    /// Indices of sampled transitions
    pub ix_sample: Option<Vec<usize>>,
}

impl<O, A> TransitionBatch for GenericTransitionBatch<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    type ObsBatch = O;
    type ActBatch = A;

    /// Decomposes the batch into its individual components.
    ///
    /// # Returns
    ///
    /// A tuple containing the following elements:
    /// 1. Observations
    /// 2. Actions
    /// 3. Next observations
    /// 4. Rewards
    /// 5. Termination flags
    /// 6. Truncation flags
    /// 7. Sample indices
    /// 8. Weights
    fn unpack(
        self,
    ) -> (
        Self::ObsBatch,
        Self::ActBatch,
        Self::ObsBatch,
        Vec<f32>,
        Vec<i8>,
        Vec<i8>,
        Option<Vec<usize>>,
        Option<Vec<f32>>,
    ) {
        (
            self.obs,
            self.act,
            self.next_obs,
            self.reward,
            self.is_terminated,
            self.is_truncated,
            self.ix_sample,
            self.weight,
        )
    }

    /// Returns the number of transitions in the batch.
    fn len(&self) -> usize {
        self.reward.len()
    }

    /// Returns a reference to the batch of observations.
    fn obs(&self) -> &Self::ObsBatch {
        &self.obs
    }

    /// Returns a reference to the batch of actions.
    fn act(&self) -> &Self::ActBatch {
        &self.act
    }
}

impl<O, A> GenericTransitionBatch<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    /// Creates a new batch with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity of the batch
    ///
    /// # Returns
    ///
    /// A new `GenericTransitionBatch` instance
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: Vec::with_capacity(capacity),
            is_terminated: Vec::with_capacity(capacity),
            is_truncated: Vec::with_capacity(capacity),
            weight: None,
            ix_sample: None,
        }
    }
}
