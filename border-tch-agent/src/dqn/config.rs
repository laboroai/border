//! Configuration of DQN agent.
use super::{
    explorer::{DQNExplorer, Softmax},
    DQNModelConfig,
};
use crate::{model::SubModel, util::OutDim, Device};
use anyhow::Result;
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};
use tch::Tensor;

#[allow(clippy::upper_case_acronyms)]
/// Constructs [DQN](super::DQN).
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct DQNConfig<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    pub(super) model_config: DQNModelConfig<Q::Config>,
    pub(super) soft_update_interval: usize,
    pub(super) n_updates_per_opt: usize,
    pub(super) min_transitions_warmup: usize,
    pub(super) batch_size: usize,
    pub(super) discount_factor: f64,
    pub(super) tau: f64,
    pub(super) train: bool,
    pub(super) explorer: DQNExplorer,
    #[serde(default)]
    pub(super) clip_reward: Option<f64>,
    #[serde(default)]
    pub(super) double_dqn: bool,
    pub(super) clip_td_err: Option<(f64, f64)>,
    pub device: Option<Device>,
    phantom: PhantomData<Q>,
}

impl<Q> Clone for DQNConfig<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
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
            train: self.train,
            explorer: self.explorer.clone(),
            clip_reward: self.clip_reward,
            double_dqn: self.double_dqn,
            clip_td_err: self.clip_td_err,
            device: self.device.clone(),
            phantom: PhantomData,    
        }
    }
}

impl<Q> Default for DQNConfig<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    /// Constructs DQN builder with default parameters.
    fn default() -> Self {
        Self {
            model_config: Default::default(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            train: false,
            // replay_burffer_capacity: 100,
            explorer: DQNExplorer::Softmax(Softmax::new()),
            // expr_sampling: ExperienceSampling::Uniform,
            clip_reward: None,
            double_dqn: false,
            clip_td_err: None,
            device: None,
            phantom: PhantomData,
        }
    }
}

impl<Q> DQNConfig<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    /// Sets soft update interval.
    pub fn soft_update_interval(mut self, v: usize) -> Self {
        self.soft_update_interval = v;
        self
    }

    /// Sets the numper of parameter update steps per optimization step.
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

    /// Explorer.
    pub fn explorer(mut self, v: DQNExplorer) -> Self {
        self.explorer = v;
        self
    }

    /// Sets the configuration of the model.
    pub fn model_config(mut self, model_config: DQNModelConfig<Q::Config>) -> Self {
        self.model_config = model_config;
        self
    }

    /// Sets the output dimention of the dqn model of the DQN agent.
    pub fn out_dim(mut self, out_dim: i64) -> Self {
        let model_config = self.model_config.clone();
        self.model_config = model_config.out_dim(out_dim);
        self
    }

    /// Reward clipping.
    pub fn clip_reward(mut self, clip_reward: Option<f64>) -> Self {
        self.clip_reward = clip_reward;
        self
    }

    /// Double DQN
    pub fn double_dqn(mut self, double_dqn: bool) -> Self {
        self.double_dqn = double_dqn;
        self
    }

    /// TD-error clipping.
    pub fn clip_td_err(mut self, clip_td_err: Option<(f64, f64)>) -> Self {
        self.clip_td_err = clip_td_err;
        self
    }

    /// Device.
    pub fn device(mut self, device: tch::Device) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Loads [DQNConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ = path.as_ref().to_owned();
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        info!("Load config of DQN agent from {}", path_.to_str().unwrap());
        Ok(b)
    }

    /// Saves [DQNConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ = path.as_ref().to_owned();
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        info!("Save config of DQN agent into {}", path_.to_str().unwrap());
        Ok(())
    }
}
