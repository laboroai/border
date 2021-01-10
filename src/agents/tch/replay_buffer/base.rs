use std::marker::PhantomData;
use tch::{Tensor, kind::{FLOAT_CPU, INT64_CPU}};
use crate::core::{Env};

pub trait TchBuffer {
    type Item;
    type SubBatch;

    fn new(capacity: usize, n_procs: usize) -> Self;

    /// Push a sample of an item (observations or actions).
    /// Note that each item may consists of values from multiple environments.
    fn push(&mut self, index: i64, item: &Self::Item);

    fn batch(&self, batch_indexes: &Tensor) -> Self::SubBatch;
}

pub struct TchBatch<E: Env, O, A> where
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act> {

    pub obs: O::SubBatch,
    pub next_obs: O::SubBatch,
    pub actions: A::SubBatch,
    pub rewards: Tensor,
    pub not_dones: Tensor,
    pub returns: Option<Tensor>,
    phantom: PhantomData<E>,
}

pub struct ReplayBuffer<E, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act> {

    obs: O,
    next_obs: O,
    actions: A,
    rewards: Tensor,
    not_dones: Tensor,
    returns: Option<Tensor>,
    capacity: usize,
    len: usize,
    i: usize,
    phandom: PhantomData<E>,
}

#[allow(clippy::len_without_is_empty)]
impl<E, O, A> ReplayBuffer<E, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act> {

    pub fn new(capacity: usize, n_procs: usize) -> Self {
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
            phandom: PhantomData,
        }
    }

    pub fn push(&mut self, obs: &O::Item, act: &A::Item, reward: &Tensor, next_obs: &O::Item,
                not_done: &Tensor) {
        let i = (self.i % self.capacity) as i64;
        self.obs.push(i, obs);
        self.next_obs.push(i, next_obs);
        self.actions.push(i, act);
        self.rewards.get(i as _).copy_(reward);
        self.not_dones.get(i as _).copy_(not_done);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn random_batch(&self, batch_size: usize) -> Option<TchBatch<E, O, A>> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let obs = self.obs.batch(&batch_indexes);
        let next_obs = self.next_obs.batch(&batch_indexes);
        let actions = self.actions.batch(&batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes).flatten(0, 1);
        let not_dones = self.not_dones.index_select(0, &batch_indexes).flatten(0, 1);
        let returns = match self.returns.as_ref() {
            Some(r) => Some(r.index_select(0, &batch_indexes).flatten(0, 1)),
            None => None
        };
        let phantom = PhantomData;

        Some(TchBatch {obs, actions, rewards, next_obs, not_dones, returns, phantom})
    }

    pub fn update_returns(&mut self, estimated_return: Tensor, gamma: f64) {
        // adapted from ppo.rs in tch-rs RL example
        self.returns = {
            let r = Tensor::zeros(&[self.len as i64], FLOAT_CPU);
            r.get(-1).unsqueeze(-1).copy_(&estimated_return);
            for s in (0..(self.len - 2) as i64).rev() {
                let r_s = self.rewards.get(s) + r.get(s + 1) * self.not_dones.get(s) * gamma;
                r.get(s).unsqueeze(-1).copy_(&r_s);
            }
            Some(r)
        };
    }

    pub fn clear_returns(&mut self) {
        self.returns = None;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
