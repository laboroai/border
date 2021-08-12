//! Replay buffer.
use border_core::Env;
use log::{info, trace};
use std::marker::PhantomData;
use tch::Tensor;

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
    fn new(capacity: usize, model_device: tch::Device) -> Self;

    /// Push a sample of an item (observations or actions).
    /// Note that each item may consists of values from multiple environments.
    fn push(&mut self, index: i64, item: &Self::Item);

    /// Constructs a batch.
    fn batch(&self, batch_indexes: &Tensor) -> Self::SubBatch;
}

/// Generic buffer on a device.
pub trait TchBufferOnDevice: TchBuffer {
    /// Constructs a [TchBuffer] on a device.
    fn new_on_device(capacity: usize, device: tch::Device, model_device: tch::Device) -> Self;
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
    pub fn new(capacity: usize, model_device: tch::Device) -> Self {
        info!("Construct replay buffer with capacity = {}", capacity);
        let capacity = capacity;

        let float_type = (tch::Kind::Float, tch::Device::Cpu);

        Self {
            obs: O::new(capacity, model_device),
            next_obs: O::new(capacity, model_device),
            actions: A::new(capacity, model_device),
            rewards: Tensor::zeros(&[capacity as _], float_type),
            not_dones: Tensor::zeros(&[capacity as _], float_type),
            returns: None,
            capacity,
            len: 0,
            i: 0,
            nonzero_reward_as_done: false,
            phandom: PhantomData,
        }
    }

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

            // TODO: Consider removing this block
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

    /// Constructs random samples.
    pub fn random_batch(&self, batch_size: usize) -> Option<TchBatch<E, O, A>> {
        let batch_size = batch_size.min(self.len - 1);
        let _no_grad = tch::no_grad_guard();
        let batch_indexes = Tensor::randint(
            (self.len - 2) as _,
            &[batch_size as _],
            (tch::Kind::Int64, self.rewards.device()),
        );
        let obs = self.obs.batch(&batch_indexes);
        let next_obs = self.next_obs.batch(&batch_indexes);
        let actions = self.actions.batch(&batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes).unsqueeze(-1);
        let not_dones = self.not_dones.index_select(0, &batch_indexes).unsqueeze(-1);
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

impl<E, O, A> ReplayBuffer<E, O, A>
where
    E: Env,
    O: TchBufferOnDevice<Item = E::Obs>,
    A: TchBufferOnDevice<Item = E::Act>,
{
    /// Constructs a replay buffer on a `device`.
    ///
    /// When generating a minibatch, the data is transferred to `model_device`.
    pub fn new_on_device(capacity: usize, device: tch::Device, model_device: tch::Device) -> Self {
        info!("Construct replay buffer with capacity = {}", capacity);
        let capacity = capacity;

        let float_type = (tch::Kind::Float, device);

        Self {
            obs: O::new_on_device(capacity, device, model_device),
            next_obs: O::new_on_device(capacity, device, model_device),
            actions: A::new_on_device(capacity, device, model_device),
            rewards: Tensor::zeros(&[capacity as _], float_type).to(device),
            not_dones: Tensor::zeros(&[capacity as _], float_type).to(device),
            returns: None,
            capacity,
            len: 0,
            i: 0,
            nonzero_reward_as_done: false,
            phandom: PhantomData,
        }
    }
}
