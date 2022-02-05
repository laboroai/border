//! Configuration of IQN agent.
use super::{IqnModelConfig, IqnSample};
use crate::{
    iqn::{EpsilonGreedy, IqnExplorer},
    model::SubModel,
    util::OutDim,
    Device,
};
use anyhow::Result;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};

#[derive(Debug, Deserialize, Serialize, PartialEq)]
/// Configuration of [Iqn](super::Iqn) agent.
pub struct IqnConfig<F, M>
where
    F: SubModel,
    M: SubModel,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
{
    pub(super) model_config: IqnModelConfig<F::Config, M::Config>,
    pub(super) soft_update_interval: usize,
    pub(super) n_updates_per_opt: usize,
    pub(super) min_transitions_warmup: usize,
    pub(super) batch_size: usize,
    pub(super) discount_factor: f64,
    pub(super) tau: f64,
    pub(super) train: bool,
    pub(super) explorer: IqnExplorer,
    pub(super) sample_percents_pred: IqnSample,
    pub(super) sample_percents_tgt: IqnSample,
    pub(super) sample_percents_act: IqnSample,
    pub device: Option<Device>,
    phantom: PhantomData<(F, M)>,
}

impl<F, M> Default for IqnConfig<F, M>
where
    F: SubModel,
    M: SubModel,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
{
    fn default() -> Self {
        Self {
            model_config: Default::default(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            sample_percents_pred: IqnSample::Uniform64,
            sample_percents_tgt: IqnSample::Uniform64,
            sample_percents_act: IqnSample::Uniform32, // Const10,
            train: false,
            explorer: IqnExplorer::EpsilonGreedy(EpsilonGreedy::default()),
            device: None,
            phantom: PhantomData,
        }
    }
}

impl<F, M> IqnConfig<F, M>
where
    F: SubModel,
    M: SubModel,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
{
    /// Sets the configuration of the model.
    pub fn model_config(mut self, model_config: IqnModelConfig<F::Config, M::Config>) -> Self {
        self.model_config = model_config;
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
    pub fn explorer(mut self, v: IqnExplorer) -> Self {
        self.explorer = v;
        self
    }

    /// Sets the output dimention of the iqn model.
    pub fn out_dim(mut self, out_dim: i64) -> Self {
        let model_config = self.model_config.clone();
        self.model_config = model_config.out_dim(out_dim);
        self
    }

    /// Sampling percent points.
    pub fn sample_percent_pred(mut self, v: IqnSample) -> Self {
        self.sample_percents_pred = v;
        self
    }

    /// Sampling percent points.
    pub fn sample_percent_tgt(mut self, v: IqnSample) -> Self {
        self.sample_percents_tgt = v;
        self
    }

    /// Sampling percent points.
    pub fn sample_percent_act(mut self, v: IqnSample) -> Self {
        self.sample_percents_act = v;
        self
    }

    /// Device.
    pub fn device(mut self, device: tch::Device) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Constructs [IqnConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [IqnConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    // /// Constructs [IQN] agent with the given replay buffer.
    // pub fn build_with_replay_bufferbuild<E, F, M, O, A>(
    //     self,
    //     iqn_model: IQNModel<F, M>,
    //     replay_buffer: ReplayBuffer<E, O, A>,
    //     expr_sampling: ExperienceSampling,
    //     device: Device,
    // ) -> IQN<E, F, M, O, A>
    // where
    //     E: Env,
    //     F: SubModel<Output = Tensor>,
    //     M: SubModel<Input = Tensor, Output = Tensor>,
    //     E::Obs: Into<F::Input>,
    //     E::Act: From<Tensor>,
    //     O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    //     A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
    // {
    //     let iqn = iqn_model;
    //     let iqn_tgt = iqn.clone();

    //     IQN {
    //         iqn,
    //         iqn_tgt,
    //         replay_buffer,
    //         prev_obs: RefCell::new(None),
    //         opt_interval_counter: self.opt_interval_counter,
    //         soft_update_interval: self.soft_update_interval,
    //         soft_update_counter: 0,
    //         n_updates_per_opt: self.n_updates_per_opt,
    //         min_transitions_warmup: self.min_transitions_warmup,
    //         batch_size: self.batch_size,
    //         discount_factor: self.discount_factor,
    //         tau: self.tau,
    //         sample_percents_pred: self.sample_percents_pred,
    //         sample_percents_tgt: self.sample_percents_tgt,
    //         sample_percents_act: self.sample_percents_act,
    //         train: self.train,
    //         explorer: self.explorer,
    //         // expr_sampling,
    //         device,
    //         phantom: PhantomData,
    //     }
    // }
}

impl<F, M> Clone for IqnConfig<F, M>
where
    F: SubModel,
    M: SubModel,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
{
    fn clone(&self) -> Self {
        Self {
            model_config: self.model_config.clone(),
            soft_update_interval: self.soft_update_interval,
            n_updates_per_opt: self.n_updates_per_opt,
            min_transitions_warmup: self.min_transitions_warmup,
            batch_size: self.batch_size,
            discount_factor: self.discount_factor,
            tau: self.tau,
            sample_percents_pred: self.sample_percents_pred.clone(),
            sample_percents_tgt: self.sample_percents_tgt.clone(),
            sample_percents_act: self.sample_percents_act.clone(),
            train: self.train,
            explorer: self.explorer.clone(),
            device: self.device.clone(),
            phantom: PhantomData,
        }
    }
}
