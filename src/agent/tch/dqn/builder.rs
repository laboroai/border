//! DQN builder.
// use log::trace;
use std::{cell::RefCell, marker::PhantomData};
use tch::Tensor;

use crate::{
    core::Env,
    agent::{
        OptInterval, OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer,
            model::Model1,
            dqn::{DQN, explorer::{DQNExplorer, Softmax}}
        }
    }
};

/// DQN builder.
pub struct DQNBuilder<E, M, O, A, Ex> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
    Ex: DQNExplorer<M>
{
    opt_interval_counter: OptIntervalCounter,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    train: bool,
    discount_factor: f64,
    tau: f64,
    explorer: Ex,
    phantom: PhantomData<(E, M, O, A, Ex)>,
}

#[allow(clippy::new_without_default)]
impl<E, M, O, A> DQNBuilder<E, M, O, A, Softmax<M>> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    /// Constructs DQN builder with default parameters.
    pub fn new() -> Self {
        Self {
            opt_interval_counter: OptInterval::Steps(1).counter(),
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            train: false,
            explorer: Softmax::new(),
            phantom: PhantomData,
        }
    }
}

impl<E, M, O, A, Ex> DQNBuilder<E, M, O, A, Ex> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
    Ex: DQNExplorer<M> + Clone
{
    /// Set optimization interval.
    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval_counter = v.counter();
        self
    }

    /// Set numper of parameter update steps per optimization step.
    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    /// Interval before starting optimization.
    pub fn min_transitions_warmup(mut self, v: usize) -> Self {
        self.min_transitions_warmup = v;
        self
    }

    /// Batch size.
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    /// Discount factor.
    pub fn discount_factor(mut self, v: f64) -> Self {
        self.discount_factor = v;
        self
    }

    /// Soft update coefficient.
    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    /// Set explorer.
    pub fn explorer<Ex2: DQNExplorer<M>>(self, v: Ex2) -> DQNBuilder<E, M, O, A, Ex2> {
        DQNBuilder {
            opt_interval_counter: self.opt_interval_counter,
            n_updates_per_opt: self.n_updates_per_opt,
            min_transitions_warmup: self.min_transitions_warmup,
            batch_size: self.batch_size,
            discount_factor: self.discount_factor,
            tau: self.tau,
            train: self.train,
            explorer: v,
            phantom: PhantomData,
        }
    }

    /// Constructs DQN.
    pub fn build(self, qnet: M, replay_buffer: ReplayBuffer<E, O, A>) -> DQN<E, M, O, A> {
        let qnet_tgt = qnet.clone();
        let explorer = Box::new(self.explorer.clone());

        DQN {
            qnet,
            qnet_tgt,
            replay_buffer,
            prev_obs: RefCell::new(None),
            opt_interval_counter: self.opt_interval_counter,
            n_updates_per_opt: self.n_updates_per_opt,
            min_transitions_warmup: self.min_transitions_warmup,
            batch_size: self.batch_size,
            discount_factor: self.discount_factor,
            tau: self.tau,
            train: self.train,
            explorer,
            phantom: PhantomData,
        }
    }
}
