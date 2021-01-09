use std::marker::PhantomData;
use tch::{Tensor, kind::{FLOAT_CPU, INT64_CPU}};
use crate::core::Env;
use crate::agents::tch::replay_buffer::{TchReplayBufferBase, TchBuffer, TchBatch};

pub trait WithCapacityAndProcs {
    fn new(capacity: usize, n_procs: usize) -> Self;
}

pub struct VecReplayBuffer<E, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs> + WithCapacityAndProcs,
    A: TchBuffer<Item = E::Act> + WithCapacityAndProcs
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
    phandom: PhantomData<E>,
}

#[allow(clippy::len_without_is_empty)]
impl<E, O, A> VecReplayBuffer<E, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs> + WithCapacityAndProcs,
    A: TchBuffer<Item = E::Act> + WithCapacityAndProcs
{
    pub fn new(capacity: usize, n_procs: usize) -> Self {
        assert!(capacity % n_procs == 0); // TODO: better error handling

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
}

impl<E, O, A> TchReplayBufferBase<E, O, A> for VecReplayBuffer<E, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs> + WithCapacityAndProcs,
    A: TchBuffer<Item = E::Act> + WithCapacityAndProcs
{
    fn push(&mut self, obs: &O::Item, action: &A::Item, reward: &Tensor, next_obs: &O::Item,
                not_done: &Tensor) {
        let i = (self.i % self.capacity) as i64;
        self.obs.push(i, obs);
        self.next_obs.push(i, next_obs);
        self.actions.push(i, action);
        self.rewards.get(i as _).copy_(reward);
        self.not_dones.get(i as _).copy_(not_done);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    fn random_batch(&self, batch_size: usize) -> Option<TchBatch<E, O, A>> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let obs = self.obs.batch(&batch_indexes);
        let next_obs = self.next_obs.batch(&batch_indexes);
        let actions = self.actions.batch(&batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);
        println!("{:?}", rewards.size());
        panic!();
        let not_dones = self.not_dones.index_select(0, &batch_indexes);
        // let returns = match self.returns.as_ref() {
        //     Some(r) => Some(r.index_select(0, &batch_indexes)),
        //     None => None
        // };
        let returns = None;
        let phantom = PhantomData;

        Some(TchBatch {obs, actions, rewards, next_obs, not_dones, returns, phantom})
    }

    fn update_returns(&mut self, estimated_return: Tensor, gamma: f64) {
        unimplemented!();
    }

    fn clear_returns(&mut self) {
        unimplemented!();
    }

    fn len(&self) -> usize {
        self.len
    }
}
