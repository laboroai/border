//! DQN builder.
// use log::trace;
use std::{cell::RefCell, marker::PhantomData};
use tch::Tensor;

use crate::{
    agent::{
        tch::{
            dqn::{
                explorer::{DQNExplorer, Softmax},
                DQN,
            },
            model::Model1,
            ReplayBuffer, TchBuffer,
        },
        OptInterval, OptIntervalCounter,
    },
    core::Env,
};

#[allow(clippy::upper_case_acronyms)]
/// DQN builder.
pub struct DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    opt_interval_counter: OptIntervalCounter,
    soft_update_interval: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    train: bool,
    discount_factor: f64,
    tau: f64,
    explorer: DQNExplorer,
    phantom: PhantomData<(E, M, O, A)>,
}

#[allow(clippy::new_without_default)]
impl<E, M, O, A> DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    /// Constructs DQN builder with default parameters.
    pub fn new() -> Self {
        Self {
            opt_interval_counter: OptInterval::Steps(1).counter(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            train: false,
            explorer: DQNExplorer::Softmax(Softmax::new()),
            phantom: PhantomData,
        }
    }
}

impl<E, M, O, A> DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    /// Set optimization interval.
    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval_counter = v.counter();
        self
    }

    /// Set soft update interval.
    pub fn soft_update_interval(mut self, v: usize) -> Self {
        self.soft_update_interval = v;
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
    pub fn explorer(mut self, v: DQNExplorer) -> DQNBuilder<E, M, O, A> where {
        self.explorer = v;
        self
    }

    /// Constructs DQN.
    pub fn build(
        self,
        qnet: M,
        replay_buffer: ReplayBuffer<E, O, A>,
        device: tch::Device,
    ) -> DQN<E, M, O, A> {
        let qnet_tgt = qnet.clone();

        DQN {
            qnet,
            qnet_tgt,
            replay_buffer,
            prev_obs: RefCell::new(None),
            opt_interval_counter: self.opt_interval_counter,
            soft_update_interval: self.soft_update_interval,
            soft_update_counter: 0,
            n_updates_per_opt: self.n_updates_per_opt,
            min_transitions_warmup: self.min_transitions_warmup,
            batch_size: self.batch_size,
            discount_factor: self.discount_factor,
            tau: self.tau,
            train: self.train,
            explorer: self.explorer,
            device,
            phantom: PhantomData,
        }
    }
}
