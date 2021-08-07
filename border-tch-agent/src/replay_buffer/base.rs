//! Replay buffer.
use super::sum_tree::SumTree;
use crate::replay_buffer::ExperienceSampling;
use border_core::Env;
use log::{info, trace};
use std::marker::PhantomData;
use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    Tensor,
};

// /// Return binary tensor, one where reward is not zero.
// ///
// /// TODO: Add test.
// fn zero_reward(reward: &Tensor) -> Tensor {
//     let zero_reward = Vec::<f32>::from(reward)
//         .iter()
//         .map(|x| if *x == 0f32 { 1f32 } else { 0f32 })
//         .collect::<Vec<_>>();
//     Tensor::of_slice(&zero_reward)
// }

/// Generic buffer inside a replay buffer.
pub trait TchBuffer {
    /// Item of the buffer.
    type Item;
    /// Batch of the items in the buffer.
    type SubBatch;

    /// Constructs a [TchBuffer].
    fn new(capacity: usize) -> Self;

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
    /// Cumulative rewards in an episode (deprecated).
    pub returns: Option<Tensor>,
    /// Indices of samples in the replay buffer.
    pub indices: Option<Tensor>,
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
    nonzero_reward_as_done: bool,
    sum_tree: Option<SumTree>,
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
    pub fn new(capacity: usize, sampling: &ExperienceSampling) -> Self {
        info!("Construct replay buffer with capacity = {}", capacity);
        let capacity = capacity;
        let sum_tree = match sampling {
            ExperienceSampling::Uniform => None,
            ExperienceSampling::TDerror { alpha } => Some(SumTree::new(capacity)),
        };

        Self {
            obs: O::new(capacity),
            next_obs: O::new(capacity),
            actions: A::new(capacity),
            rewards: Tensor::zeros(&[capacity as _], FLOAT_CPU),
            not_dones: Tensor::zeros(&[capacity as _], FLOAT_CPU),
            returns: None,
            capacity,
            len: 0,
            i: 0,
            nonzero_reward_as_done: false,
            sum_tree: None,
            phandom: PhantomData,
        }
    }

    // /// If set to `True`, non-zero reward is considered as the end of episodes.
    // #[deprecated]
    // pub fn nonzero_reward_as_done(mut self, _v: bool) -> Self {
    //     unimplemented!();
    //     // self.nonzero_reward_as_done = v;
    //     // self
    // }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
        self.i = 0;
        self.returns = None;
    }

    /// Pushes a tuple of observation, action, reward, next observation and is_done flag.
    ///
    /// Both `reward.size().len()` and `not_done().len()` must be 1; these are vectors.
    pub fn push(
        &mut self,
        obs: &O::Item,
        act: &A::Item,
        reward: &Tensor,
        next_obs: &O::Item,
        not_done: &Tensor,
    ) {
        trace!("ReplayBuffer::push()");

        // Push a minibatch at once
        self.obs.push(self.i as _, obs);
        self.next_obs.push(self.i as _, next_obs);
        self.actions.push(self.i as _, act);

        // Loop over minibatch
        let batch_size = reward.size()[0];
        for j in 0..batch_size {
            self.rewards.get(self.i as _).copy_(&reward.get(j));

            if !self.nonzero_reward_as_done {
                self.not_dones.get(self.i as _).copy_(&not_done.get(j));
            } else {
                unimplemented!();
                // let zero_reward = zero_reward(reward);
                // self.not_dones
                //     .get(i as _)
                //     .copy_(&(zero_reward * not_done).unsqueeze(-1));
            }

            self.i = (self.i + 1) % self.capacity;
            if self.len < self.capacity {
                self.len += 1;
            }
        }
    }

    /// Take samples for creating minibatch.
    fn sampling(&self, batch_size: usize) -> Tensor {
        match &self.sum_tree {
            None => {
                let batch_size = batch_size.min(self.len - 1);
                Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU)        
            }
            Some(sum_tree) => {
                panic!();
            }
        }
    }

    /// Constructs random samples.
    pub fn random_batch(&self, batch_size: usize) -> Option<TchBatch<E, O, A>> {
        let batch_indexes = self.sampling(batch_size);
        let obs = self.obs.batch(&batch_indexes);
        let next_obs = self.next_obs.batch(&batch_indexes);
        let actions = self.actions.batch(&batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes).unsqueeze(-1); //.flatten(0, 1);
        let not_dones = self.not_dones.index_select(0, &batch_indexes).unsqueeze(-1); //.flatten(0, 1);
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
            indices: Some(batch_indexes),
            phantom,
        })
    }

    /// Updates priority of samples in the buffer.
    pub fn update_priority(&mut self, indices: &Tensor, p: &Tensor) {
        if self.sum_tree.is_none() {
            panic!();
        }
    }

    /// Updates returns in the replay buffer.
    #[deprecated]
    pub fn update_returns(&mut self, _estimated_return: Tensor, _gamma: f64) {
        unimplemented!();
        // trace!("Start update returns");

        // // adapted from ppo.rs in tch-rs RL example
        // self.returns = {
        //     let r = Tensor::zeros(&[self.len as _, self.n_procs as _], FLOAT_CPU);
        //     trace!("r.shape                = {:?}", r.size());
        //     trace!("estimated_return.shape = {:?}", estimated_return.size());

        //     r.get((self.len - 1) as _)
        //         .copy_(&estimated_return.squeeze());
        //     trace!("Set estimated_return to the tail of the buffer");

        //     for s in (0..(self.len - 2) as i64).rev() {
        //         trace!(
        //             "self.rewards.get(s).shape   = {:?}",
        //             self.rewards.get(s).size()
        //         );
        //         trace!("r.get(s).shape              = {:?}", r.get(s).size());
        //         trace!(
        //             "self.not_dones.get(s).shape = {:?}",
        //             self.not_dones.get(s).size()
        //         );

        //         let r_s = self.rewards.get(s).squeeze()
        //             + gamma * r.get(s + 1) * self.not_dones.get(s).squeeze();
        //         trace!("Compute r_s");
        //         trace!("r_s.shape = {:?}", r_s.size());
        //         r.get(s).copy_(&r_s);
        //         trace!("Set r_s to the return buffer");
        //     }
        //     Some(r)
        // };
    }

    /// Clears returns.
    #[deprecated]
    pub fn clear_returns(&mut self) {
        unimplemented!();
        // self.returns = None;
    }

    /// Length of the replay buffer.
    pub fn len(&self) -> usize {
        self.len
    }
}
