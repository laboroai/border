use std::marker::PhantomData;
use tch::{Tensor, kind::{FLOAT_CPU, INT64_CPU}};
use crate::core::{Env};

pub trait TchBufferableObsInfo {
    fn tch_kind() -> (tch::Kind, tch::Device);

    fn shape() -> Vec<i64>;
}

pub trait TchBufferableActInfo {
    fn tch_kind() -> (tch::Kind, tch::Device);

    fn shape() -> Vec<i64>;
}

pub struct ReplayBuffer<E> where
    E: Env,
    E::Obs: TchBufferableObsInfo,
    E::Act: TchBufferableActInfo {
    obs: Tensor,
    next_obs: Tensor,
    rewards: Tensor,
    actions: Tensor,
    not_dones: Tensor,
    returns: Option<Tensor>,
    capacity: usize,
    len: usize,
    i: usize,
    phandom: PhantomData<E>,
}

fn concat(capacity: usize, shape: &[i64]) -> Vec<i64> {
    [&[capacity as i64], shape].concat()
}

pub struct Batch {
    pub obs: Tensor,
    pub next_obs: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub not_dones: Tensor,
}

pub struct Batch2 {
    pub obs: Tensor,
    pub next_obs: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub not_dones: Tensor,
    pub returns: Tensor,
}

#[allow(clippy::len_without_is_empty)]
impl<E> ReplayBuffer<E> where
    E: Env,
    E::Obs: TchBufferableObsInfo,
    E::Act: TchBufferableActInfo {
    pub fn new(capacity: usize) -> Self {
        let shape_obs = concat(capacity, &E::Obs::shape().as_slice());
        let shape_act = concat(capacity, &E::Act::shape().as_slice());
        // TODO: choose kind of action (FLOAT_CPU or INT64_CPU) depending on
        // whether action is discrete or continuous
        Self {
            obs: Tensor::zeros(shape_obs.as_slice(), E::Obs::tch_kind()),
            next_obs: Tensor::zeros(shape_obs.as_slice(), E::Obs::tch_kind()),
            rewards: Tensor::zeros(&[capacity as _, 1], FLOAT_CPU),
            actions: Tensor::zeros(shape_act.as_slice(), E::Act::tch_kind()),
            not_dones: Tensor::zeros(&[capacity as _, 1], FLOAT_CPU),
            returns: None,
            capacity,
            len: 0,
            i: 0,
            phandom: PhantomData,
        }
    }

    pub fn push(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor,
                not_done: &Tensor) {
        let i = self.i % self.capacity;
        self.obs.get(i as _).copy_(obs);
        self.rewards.get(i as _).copy_(reward);
        self.actions.get(i as _).copy_(actions);
        self.next_obs.get(i as _).copy_(next_obs);
        self.not_dones.get(i as _).copy_(not_done);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn random_batch(&self, batch_size: usize) -> Option<Batch> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let obs = self.obs.index_select(0, &batch_indexes);
        let next_obs = self.next_obs.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);
        let not_dones = self.not_dones.index_select(0, &batch_indexes);

        Some(Batch {obs, actions, rewards, next_obs, not_dones})
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

    pub fn random_batch2(&self, batch_size: usize) -> Option<Batch2> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let obs = self.obs.index_select(0, &batch_indexes);
        let next_obs = self.next_obs.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);
        let not_dones = self.not_dones.index_select(0, &batch_indexes);
        let returns = self.returns.as_ref()?.index_select(0, &batch_indexes);

        Some(Batch2 {obs, actions, rewards, next_obs, not_dones, returns})
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
