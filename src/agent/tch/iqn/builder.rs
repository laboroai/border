//! Builder of IQN agent
use std::{cell::RefCell, marker::PhantomData, default::Default};
use tch::{Tensor, Device};

use crate::{
    core::Env,
    agent::{
        OptInterval, OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer, model::SubModel,
            IQN, IQNModel, IQNExplorer, EpsilonGreedy
        }
    }
};

#[allow(clippy::clippy::upper_case_acronyms)]
/// IQN builder.
pub struct IQNBuilder<E, F, M, O, A> where
    E: Env,
    F: SubModel,
    M: SubModel,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    opt_interval_counter: OptIntervalCounter,
    soft_update_interval: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    discount_factor: f64,
    tau: f64,
    n_prob_samples: usize,
    train: bool,
    explorer: IQNExplorer,
    phantom: PhantomData<(E, F, M, O, A)>,
}

impl<E, F, M, O, A> Default for IQNBuilder<E, F, M, O, A> where
    E: Env,
    F: SubModel,
    M: SubModel,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn default() -> Self {
        Self {
            opt_interval_counter: OptInterval::Steps(1).counter(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            n_prob_samples: 10, // IQNSample::Uniform10
            train: false,
            explorer: IQNExplorer::EpsilonGreedy(EpsilonGreedy::new()),
            phantom: PhantomData,
        }
    }
}

impl<E, F, M, O, A> IQNBuilder<E, F, M, O, A> where
    E: Env,
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
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
    pub fn explorer(mut self, v: IQNExplorer) -> Self where
    {
        self.explorer = v;
        self
    }
    
    /// Constructs [IQN] agent.
    pub fn build(self, iqn_model: IQNModel<F, M>, replay_buffer: ReplayBuffer<E, O, A>, device: Device)
        -> IQN<E, F, M, O, A> 
    {
        let iqn = iqn_model;
        let iqn_tgt = iqn.clone();

        IQN {
            iqn,
            iqn_tgt,
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
            n_prob_samples: self.n_prob_samples,
            train: self.train,
            explorer: self.explorer,
            device,
            phantom: PhantomData,
        }
    }
}
