//! Generic implementation of replay buffers for reinforcement learning.
//!
//! This module provides a generic implementation of replay buffers that can store
//! and sample transitions of arbitrary observation and action types. It supports:
//! - Standard experience replay
//! - Prioritized experience replay (PER)
//! - Importance sampling weights for off-policy learning

mod iw_scheduler;
mod sum_tree;
use super::{config::PerConfig, BatchBase, GenericTransitionBatch, SimpleReplayBufferConfig};
use crate::{ExperienceBufferBase, ReplayBufferBase, TransitionBatch};
use anyhow::Result;
pub use iw_scheduler::IwScheduler;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sum_tree::SumTree;
pub use sum_tree::WeightNormalizer;

/// State management for Prioritized Experience Replay (PER).
///
/// This struct maintains the necessary state for PER, including:
/// - A sum tree for efficient priority sampling
/// - An importance weight scheduler for adjusting sample weights
struct PerState {
    /// A sum tree data structure for efficient priority sampling.
    sum_tree: SumTree,

    /// Scheduler for importance sampling weights.
    iw_scheduler: IwScheduler,
}

impl PerState {
    /// Creates a new PER state with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of transitions to store
    /// * `per_config` - Configuration for prioritized experience replay
    fn new(capacity: usize, per_config: &PerConfig) -> Self {
        Self {
            sum_tree: SumTree::new(capacity, per_config.alpha, per_config.normalize),
            iw_scheduler: IwScheduler::new(
                per_config.beta_0,
                per_config.beta_final,
                per_config.n_opts_final,
            ),
        }
    }
}

/// A generic implementation of a replay buffer for reinforcement learning.
///
/// This buffer can store transitions of arbitrary observation and action types,
/// making it suitable for a wide range of reinforcement learning tasks. It supports:
/// - Standard experience replay
/// - Prioritized experience replay (optional)
/// - Efficient sampling and storage
///
/// # Type Parameters
///
/// * `O` - The type of observations, must implement [`BatchBase`]
/// * `A` - The type of actions, must implement [`BatchBase`]
///
/// # Examples
///
/// ```ignore
/// let config = SimpleReplayBufferConfig {
///     capacity: 10000,
///     per_config: Some(PerConfig {
///         alpha: 0.6,
///         beta_0: 0.4,
///         beta_final: 1.0,
///         n_opts_final: 100000,
///         normalize: true,
///     }),
/// };
///
/// let mut buffer = SimpleReplayBuffer::<Tensor, Tensor>::build(&config);
///
/// // Add transitions
/// buffer.push(transition)?;
///
/// // Sample a batch
/// let batch = buffer.batch(32)?;
/// ```
pub struct SimpleReplayBuffer<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    /// Maximum number of transitions that can be stored.
    capacity: usize,

    /// Current insertion index.
    i: usize,

    /// Current number of stored transitions.
    size: usize,

    /// Storage for observations.
    obs: O,

    /// Storage for actions.
    act: A,

    /// Storage for next observations.
    next_obs: O,

    /// Storage for rewards.
    reward: Vec<f32>,

    /// Storage for termination flags.
    is_terminated: Vec<i8>,

    /// Storage for truncation flags.
    is_truncated: Vec<i8>,

    /// Random number generator for sampling.
    rng: StdRng,

    /// State for prioritized experience replay, if enabled.
    per_state: Option<PerState>,
}

impl<O, A> SimpleReplayBuffer<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    /// Pushes rewards into the buffer at the specified index.
    ///
    /// # Arguments
    ///
    /// * `i` - Starting index for insertion
    /// * `b` - Vector of rewards to insert
    #[inline]
    fn push_reward(&mut self, i: usize, b: &Vec<f32>) {
        let mut j = i;
        for r in b.iter() {
            self.reward[j] = *r;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    /// Pushes termination flags into the buffer at the specified index.
    ///
    /// # Arguments
    ///
    /// * `i` - Starting index for insertion
    /// * `b` - Vector of termination flags to insert
    #[inline]
    fn push_is_terminated(&mut self, i: usize, b: &Vec<i8>) {
        let mut j = i;
        for d in b.iter() {
            self.is_terminated[j] = *d;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    /// Pushes truncation flags into the buffer at the specified index.
    ///
    /// # Arguments
    ///
    /// * `i` - Starting index for insertion
    /// * `b` - Vector of truncation flags to insert
    fn push_is_truncated(&mut self, i: usize, b: &Vec<i8>) {
        let mut j = i;
        for d in b.iter() {
            self.is_truncated[j] = *d;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    /// Samples rewards for the given indices.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Indices to sample from
    ///
    /// # Returns
    ///
    /// Vector of sampled rewards
    fn sample_reward(&self, ixs: &Vec<usize>) -> Vec<f32> {
        ixs.iter().map(|ix| self.reward[*ix]).collect()
    }

    /// Samples termination flags for the given indices.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Indices to sample from
    ///
    /// # Returns
    ///
    /// Vector of sampled termination flags
    fn sample_is_terminated(&self, ixs: &Vec<usize>) -> Vec<i8> {
        ixs.iter().map(|ix| self.is_terminated[*ix]).collect()
    }

    /// Samples truncation flags for the given indices.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Indices to sample from
    ///
    /// # Returns
    ///
    /// Vector of sampled truncation flags
    fn sample_is_truncated(&self, ixs: &Vec<usize>) -> Vec<i8> {
        ixs.iter().map(|ix| self.is_truncated[*ix]).collect()
    }

    /// Sets priorities for newly added samples in prioritized experience replay.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of new samples to prioritize
    fn set_priority(&mut self, batch_size: usize) {
        let sum_tree = &mut self.per_state.as_mut().unwrap().sum_tree;
        let max_p = sum_tree.max();

        for j in 0..batch_size {
            let i = (self.i + j) % self.capacity;
            sum_tree.add(i, max_p);
        }
    }

    /// Returns a batch containing all actions in the buffer.
    ///
    /// # Warning
    ///
    /// This method should be used with caution on large replay buffers
    /// as it may consume significant memory.
    pub fn whole_actions(&self) -> A {
        let ixs = (0..self.size).collect::<Vec<_>>();
        self.act.sample(&ixs)
    }

    /// Returns the number of terminated episodes in the buffer.
    pub fn num_terminated_flags(&self) -> usize {
        self.is_terminated
            .iter()
            .map(|is_terminated| *is_terminated as usize)
            .sum()
    }

    /// Returns the number of truncated episodes in the buffer.
    pub fn num_truncated_flags(&self) -> usize {
        self.is_truncated
            .iter()
            .map(|is_truncated| *is_truncated as usize)
            .sum()
    }

    /// Returns the sum of all rewards in the buffer.
    pub fn sum_rewards(&self) -> f32 {
        self.reward.iter().sum()
    }
}

impl<O, A> ExperienceBufferBase for SimpleReplayBuffer<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    type Item = GenericTransitionBatch<O, A>;

    /// Returns the current number of transitions in the buffer.
    fn len(&self) -> usize {
        self.size
    }

    /// Adds a new transition to the buffer.
    ///
    /// # Arguments
    ///
    /// * `tr` - The transition to add
    ///
    /// # Returns
    ///
    /// `Ok(())` if the transition was added successfully
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is full and cannot accept more transitions
    fn push(&mut self, tr: Self::Item) -> Result<()> {
        let len = tr.len(); // batch size
        let (obs, act, next_obs, reward, is_terminated, is_truncated, _, _) = tr.unpack();
        self.obs.push(self.i, obs);
        self.act.push(self.i, act);
        self.next_obs.push(self.i, next_obs);
        self.push_reward(self.i, &reward);
        self.push_is_terminated(self.i, &is_terminated);
        self.push_is_truncated(self.i, &is_truncated);

        if self.per_state.is_some() {
            self.set_priority(len)
        };

        self.i = (self.i + len) % self.capacity;
        self.size += len;
        if self.size >= self.capacity {
            self.size = self.capacity;
        }

        Ok(())
    }
}

impl<O, A> ReplayBufferBase for SimpleReplayBuffer<O, A>
where
    O: BatchBase,
    A: BatchBase,
{
    type Config = SimpleReplayBufferConfig;
    type Batch = GenericTransitionBatch<O, A>;

    /// Creates a new replay buffer with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the replay buffer
    ///
    /// # Returns
    ///
    /// A new instance of the replay buffer
    fn build(config: &Self::Config) -> Self {
        let capacity = config.capacity;
        let per_state = match &config.per_config {
            Some(per_config) => Some(PerState::new(capacity, per_config)),
            None => None,
        };

        Self {
            capacity,
            i: 0,
            size: 0,
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: vec![0.; capacity],
            is_terminated: vec![0; capacity],
            is_truncated: vec![0; capacity],
            rng: StdRng::seed_from_u64(config.seed as _),
            per_state,
        }
    }

    /// Samples a batch of transitions from the buffer.
    ///
    /// If prioritized experience replay is enabled, samples are selected
    /// according to their priorities. Otherwise, uniform random sampling is used.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of transitions to sample
    ///
    /// # Returns
    ///
    /// A batch of sampled transitions
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer is empty
    /// - The requested batch size is larger than the buffer size
    fn batch(&mut self, size: usize) -> Result<Self::Batch> {
        let (ixs, weight) = if let Some(per_state) = &self.per_state {
            let sum_tree = &per_state.sum_tree;
            let beta = per_state.iw_scheduler.beta();
            let (ixs, weight) = sum_tree.sample(size, beta);
            let ixs = ixs.iter().map(|&ix| ix as usize).collect();
            (ixs, Some(weight))
        } else {
            let ixs = (0..size)
                // .map(|_| self.rng.usize(..self.size))
                .map(|_| (self.rng.next_u32() as usize) % self.size)
                .collect::<Vec<_>>();
            let weight = None;
            (ixs, weight)
        };

        Ok(Self::Batch {
            obs: self.obs.sample(&ixs),
            act: self.act.sample(&ixs),
            next_obs: self.next_obs.sample(&ixs),
            reward: self.sample_reward(&ixs),
            is_terminated: self.sample_is_terminated(&ixs),
            is_truncated: self.sample_is_truncated(&ixs),
            ix_sample: Some(ixs),
            weight,
        })
    }

    /// Updates the priorities of transitions in the buffer.
    ///
    /// This method is used in prioritized experience replay to adjust
    /// the sampling probabilities based on TD errors.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Optional indices of transitions to update
    /// * `td_errs` - Optional TD errors for the transitions
    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_errs: &Option<Vec<f32>>) {
        if let Some(per_state) = &mut self.per_state {
            let ixs = ixs
                .as_ref()
                .expect("ixs should be Some(_) in update_priority().");
            let td_errs = td_errs
                .as_ref()
                .expect("td_errs should be Some(_) in update_priority().");
            for (&ix, &td_err) in ixs.iter().zip(td_errs.iter()) {
                per_state.sum_tree.update(ix, td_err);
            }
            per_state.iw_scheduler.add_n_opts();
        }
    }
}
