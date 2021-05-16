//! Replay buffer.
use log::{trace, info};
use std::marker::PhantomData;
use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    Tensor,
};

use crate::core::Env;

/// Return binary tensor, one where reward is not zero.
///
/// TODO: Add test.
fn zero_reward(reward: &Tensor) -> Tensor {
    let zero_reward = Vec::<f32>::from(reward)
        .iter()
        .map(|x| if *x == 0f32 { 1f32 } else { 0f32 })
        .collect::<Vec<_>>();
    Tensor::of_slice(&zero_reward)
}

/// Generic buffer inside a replay buffer.
pub trait TchBuffer {
    /// Item of the buffer.
    type Item;
    /// Batch of the items in the buffer.
    type SubBatch;

    /// Constructs a [TchBuffer].
    fn new(capacity: usize, n_procs: usize) -> Self;

    /// Push a sample of an item (observations or actions).
    /// Note that each item may consists of values from multiple environments.
    fn push(&mut self, index: i64, item: &Self::Item);

    /// Constructs a batch.
    fn batch(&self, batch_indexes: &Tensor) -> Self::SubBatch;
}

/// Batch object, generic wrt observation and action.
pub struct TchBatch<E: Env, O, A>
where
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    /// Generic observation.
    pub obs: O::SubBatch,
    /// Generic observation at the next step.
    pub next_obs: O::SubBatch,
    /// Generic action.
    pub actions: A::SubBatch,
    /// Reward.
    pub rewards: Tensor,
    /// Flag if episode is done.
    pub not_dones: Tensor,
    /// Cumulative rewards in an episode.
    pub returns: Option<Tensor>,
    phantom: PhantomData<E>,
}

/// Replay buffer.
pub struct ReplayBuffer<E, O, A>
where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    obs: O,
    next_obs: O,
    actions: A,
    rewards: Tensor,
    not_dones: Tensor,
    returns: Option<Tensor>,
    capacity: usize,
    len: usize,
    i: usize,
    n_procs: usize,
    nonzero_reward_as_done: bool,
    phandom: PhantomData<E>,
}

#[allow(clippy::len_without_is_empty)]
impl<E, O, A> ReplayBuffer<E, O, A>
where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    /// Constructs a replay buffer.
    pub fn new(capacity: usize, n_procs: usize) -> Self {
        if capacity % n_procs != 0 {
            // TODO: Rusty error handling
            panic!("capacity % n_procs must be 0");
        }

        let capacity = capacity / n_procs;

        info!("Construct replay buffer");
        info!("Capacity       = {}", capacity);
        info!("Num. of procs  = {}", n_procs);
        info!("Total capacity = {}", capacity * n_procs);

        Self {
            obs: O::new(capacity, n_procs),
            next_obs: O::new(capacity, n_procs),
            actions: A::new(capacity, n_procs),
            rewards: Tensor::zeros(&[capacity as _, n_procs as _, 1], FLOAT_CPU),
            not_dones: Tensor::zeros(&[capacity as _, n_procs as _, 1], FLOAT_CPU),
            returns: None,
            capacity,
            len: 0,
            i: 0,
            n_procs,
            nonzero_reward_as_done: false,
            phandom: PhantomData,
        }
    }

    /// If set to `True`, non-zero reward is considered as the end of episodes.
    pub fn nonzero_reward_as_done(mut self, v: bool) -> Self {
        self.nonzero_reward_as_done = v;
        self
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
        self.i = 0;
        self.returns = None;
    }

    /// Pushes a tuple of observation, action, reward, next observation and is_done flag.
    pub fn push(
        &mut self,
        obs: &O::Item,
        act: &A::Item,
        reward: &Tensor,
        next_obs: &O::Item,
        not_done: &Tensor,
    ) {
        trace!("ReplayBuffer::push()");

        let i = (self.i % self.capacity) as i64;
        self.obs.push(i, obs);
        self.next_obs.push(i, next_obs);
        self.actions.push(i, act);
        self.rewards.get(i as _).copy_(&reward.unsqueeze(-1));

        if !self.nonzero_reward_as_done {
            self.not_dones.get(i as _).copy_(&not_done.unsqueeze(-1));
        } else {
            let zero_reward = zero_reward(reward);
            self.not_dones
                .get(i as _)
                .copy_(&(zero_reward * not_done).unsqueeze(-1));
        }

        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Constructs random samples.
    pub fn random_batch(&self, batch_size: usize) -> Option<TchBatch<E, O, A>> {
        let batch_size = batch_size.min(self.len - 1);

        if batch_size % self.n_procs != 0 {
            // TODO: Rusty error handling
            panic!("batch_size % n_procs must be 0.");
        }

        let batch_size = batch_size / self.n_procs;
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let obs = self.obs.batch(&batch_indexes);
        let next_obs = self.next_obs.batch(&batch_indexes);
        let actions = self.actions.batch(&batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes).flatten(0, 1);
        let not_dones = self.not_dones.index_select(0, &batch_indexes).flatten(0, 1);
        let returns = match self.returns.as_ref() {
            Some(r) => Some(r.index_select(0, &batch_indexes).flatten(0, 1)),
            None => None,
        };
        let phantom = PhantomData;

        Some(TchBatch {
            obs,
            actions,
            rewards,
            next_obs,
            not_dones,
            returns,
            phantom,
        })
    }

    /// Updates returns in the replay buffer.
    pub fn update_returns(&mut self, estimated_return: Tensor, gamma: f64) {
        trace!("Start update returns");

        // adapted from ppo.rs in tch-rs RL example
        self.returns = {
            let r = Tensor::zeros(&[self.len as _, self.n_procs as _], FLOAT_CPU);
            trace!("r.shape                = {:?}", r.size());
            trace!("estimated_return.shape = {:?}", estimated_return.size());

            r.get((self.len - 1) as _)
                .copy_(&estimated_return.squeeze());
            trace!("Set estimated_return to the tail of the buffer");

            for s in (0..(self.len - 2) as i64).rev() {
                trace!(
                    "self.rewards.get(s).shape   = {:?}",
                    self.rewards.get(s).size()
                );
                trace!("r.get(s).shape              = {:?}", r.get(s).size());
                trace!(
                    "self.not_dones.get(s).shape = {:?}",
                    self.not_dones.get(s).size()
                );

                let r_s = self.rewards.get(s).squeeze()
                    + gamma * r.get(s + 1) * self.not_dones.get(s).squeeze();
                trace!("Compute r_s");
                trace!("r_s.shape = {:?}", r_s.size());
                r.get(s).copy_(&r_s);
                trace!("Set r_s to the return buffer");
            }
            Some(r)
        };
    }

    /// Clears returns.
    pub fn clear_returns(&mut self) {
        self.returns = None;
    }

    /// Length of the replay buffer.
    pub fn len(&self) -> usize {
        self.len
    }
}
