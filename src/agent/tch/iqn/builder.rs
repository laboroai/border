//! Constructs IQN agent.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    default::Default,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};
use tch::{Device, Tensor};

use crate::{
    agent::{
        tch::{
            iqn::{EpsilonGreedy, IQNExplorer, IQNModel, IQN},
            model::SubModel,
            ReplayBuffer, TchBuffer,
        },
        OptInterval, OptIntervalCounter,
    },
    core::Env,
};

use super::model::IQNSample;

#[allow(clippy::clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Constructs IQN agent.
pub struct IQNBuilder {
    opt_interval_counter: OptIntervalCounter,
    soft_update_interval: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    discount_factor: f64,
    tau: f64,
    sample_percents_pred: IQNSample,
    sample_percents_tgt: IQNSample,
    sample_percents_act: IQNSample,
    train: bool,
    explorer: IQNExplorer,
}

impl Default for IQNBuilder {
    fn default() -> Self {
        Self {
            opt_interval_counter: OptInterval::Steps(1).counter(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            sample_percents_pred: IQNSample::Uniform64,
            sample_percents_tgt: IQNSample::Uniform64,
            sample_percents_act: IQNSample::Uniform32, // Const10,
            train: false,
            explorer: IQNExplorer::EpsilonGreedy(EpsilonGreedy::default()),
        }
    }
}

impl IQNBuilder {
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
    pub fn explorer(mut self, v: IQNExplorer) -> Self where {
        self.explorer = v;
        self
    }

    /// Sampling percent points.
    pub fn sample_percent_pred(mut self, v: IQNSample) -> Self {
        self.sample_percents_pred = v;
        self
    }

    /// Sampling percent points.
    pub fn sample_percent_tgt(mut self, v: IQNSample) -> Self {
        self.sample_percents_tgt = v;
        self
    }

    /// Sampling percent points.
    pub fn sample_percent_act(mut self, v: IQNSample) -> Self {
        self.sample_percents_act = v;
        self
    }

    /// Constructs [IQNBuilder] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [IQNBuilder].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    /// Constructs [IQN] agent.
    pub fn build<E, F, M, O, A>(
        self,
        iqn_model: IQNModel<F, M>,
        replay_buffer: ReplayBuffer<E, O, A>,
        device: Device,
    ) -> IQN<E, F, M, O, A>
    where
        E: Env,
        F: SubModel<Output = Tensor>,
        M: SubModel<Input = Tensor, Output = Tensor>,
        E::Obs: Into<F::Input>,
        E::Act: From<Tensor>,
        O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
        A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
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
            sample_percents_pred: self.sample_percents_pred,
            sample_percents_tgt: self.sample_percents_tgt,
            sample_percents_act: self.sample_percents_act,
            train: self.train,
            explorer: self.explorer,
            device,
            phantom: PhantomData,
        }
    }
}
