//! Types and traits for handling batches of transitions in reinforcement learning.
//!
//! This module provides abstractions for working with batches of transitions,
//! which are essential for training reinforcement learning agents. A transition
//! represents a single step in the environment, containing the observation,
//! action, next observation, reward, and termination information.

/// A batch of transitions used for training reinforcement learning agents.
///
/// This trait represents a collection of transitions in the form `(o_t, a_t, o_t+n, r_t, is_terminated_t, is_truncated_t)`,
/// where:
/// - `o_t` is the observation at time step t
/// - `a_t` is the action taken at time step t
/// - `o_t+n` is the observation n steps after t
/// - `r_t` is the reward received after taking action `a_t`
/// - `is_terminated_t` indicates if the episode terminated at this step
/// - `is_truncated_t` indicates if the episode was truncated at this step
///
/// The value of n determines the type of backup:
/// - When n = 1, it represents a standard one-step transition
/// - When n > 1, it represents an n-step transition, which can be used for
///   n-step temporal difference learning
///
/// # Associated Types
///
/// * `ObsBatch` - The type used to store batches of observations
/// * `ActBatch` - The type used to store batches of actions
///
/// # Examples
///
/// A typical use case is in Q-learning, where transitions are used to update
/// the Q-function:
/// ```ignore
/// let (obs, act, next_obs, reward, is_terminated, is_truncated, _, _) = batch.unpack();
/// let target = reward + gamma * (1 - is_terminated) * max_a Q(next_obs, a);
/// ```
pub trait TransitionBatch {
    /// The type used to store batches of observations.
    ///
    /// This type must be able to efficiently store and access multiple observations
    /// simultaneously, typically implemented as a tensor or array-like structure.
    type ObsBatch;

    /// The type used to store batches of actions.
    ///
    /// This type must be able to efficiently store and access multiple actions
    /// simultaneously, typically implemented as a tensor or array-like structure.
    type ActBatch;

    /// Unpacks the batch into its constituent parts.
    ///
    /// Returns a tuple containing:
    /// 1. The batch of observations at time t
    /// 2. The batch of actions taken at time t
    /// 3. The batch of observations at time t+n
    /// 4. The batch of rewards received
    /// 5. The batch of termination flags
    /// 6. The batch of truncation flags
    /// 7. Optional sample indices (used for prioritized experience replay)
    /// 8. Optional importance weights (used for prioritized experience replay)
    ///
    /// # Returns
    ///
    /// A tuple containing all components of the transition batch, with optional
    /// metadata for prioritized experience replay.
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
    );

    /// Returns the number of transitions in the batch.
    ///
    /// This is typically used to determine the batch size for optimization steps
    /// and to verify that all components of the batch have consistent sizes.
    fn len(&self) -> usize;

    /// Returns a reference to the batch of observations at time t.
    ///
    /// This provides efficient access to the observations without unpacking the
    /// entire batch.
    fn obs(&self) -> &Self::ObsBatch;

    /// Returns a reference to the batch of actions taken at time t.
    ///
    /// This provides efficient access to the actions without unpacking the
    /// entire batch.
    fn act(&self) -> &Self::ActBatch;
}
