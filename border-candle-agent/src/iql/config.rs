//! Configuration of IQL agent.
use super::ValueConfig;
use crate::{
    model::{SubModel1, SubModel2},
    util::{actor::GaussianActorConfig, critic::MultiCriticConfig, CriticLoss, OutDim},
    Device,
};
use anyhow::Result;
use candle_core::Tensor;
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration of [`Iql`](super::Iql).
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct IqlConfig<Q, P, V>
where
    Q: SubModel2<Output = Tensor>,
    P: SubModel1<Output = (Tensor, Tensor)>,
    V: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
{
    /// Configuration of the value model.
    pub value_config: ValueConfig<V::Config>,

    /// Configuration of the critic model.
    pub critic_config: MultiCriticConfig<Q::Config>,

    /// Configuration of the actor model.
    pub actor_config: GaussianActorConfig<P::Config>,

    /// Discont factor.
    pub gamma: f32,

    /// Expectile value.
    pub tau_iql: f64,

    /// The inverse of lambda in the paper.
    pub inv_lambda: f64,

    /// Number of parameter updates per optimization step.
    pub n_updates_per_opt: usize,

    /// Batch size for training.
    pub batch_size: usize,

    // /// Scaling factor for rewards.
    // pub reward_scale: f32,

    /// If true, advantage weights are calculated with softmax within each mini-batch.
    pub adv_softmax: bool,

    // /// If `true`, the agent is
    // pub train: bool,
    /// Type of critic loss function.
    pub critic_loss: CriticLoss,

    /// Device used for the actor and critic models (e.g., CPU or GPU).
    pub device: Option<Device>,

    /// Maximum of exponent of advantage.
    pub exp_adv_max: f64,
}

impl<Q, P, V> Clone for IqlConfig<Q, P, V>
where
    Q: SubModel2<Output = Tensor>,
    P: SubModel1<Output = (Tensor, Tensor)>,
    V: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
{
    fn clone(&self) -> Self {
        Self {
            value_config: self.value_config.clone(),
            critic_config: self.critic_config.clone(),
            actor_config: self.actor_config.clone(),
            gamma: self.gamma,
            tau_iql: self.tau_iql,
            inv_lambda: self.inv_lambda,
            n_updates_per_opt: self.n_updates_per_opt,
            batch_size: self.batch_size,
            // reward_scale: self.reward_scale,
            adv_softmax: self.adv_softmax,
            critic_loss: self.critic_loss.clone(),
            device: self.device.clone(),
            exp_adv_max: self.exp_adv_max,
        }
    }
}

impl<Q, P, V> Default for IqlConfig<Q, P, V>
where
    Q: SubModel2<Output = Tensor>,
    P: SubModel1<Output = (Tensor, Tensor)>,
    V: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
{
    fn default() -> Self {
        Self {
            value_config: Default::default(),
            critic_config: Default::default(),
            actor_config: Default::default(),
            gamma: 0.99,
            tau_iql: 0.7,
            inv_lambda: 10.0,
            n_updates_per_opt: 1,
            batch_size: 1,
            // reward_scale: 1.0,
            adv_softmax: false,
            critic_loss: CriticLoss::Mse,
            device: None,
            exp_adv_max: 100.0,
        }
    }
}

impl<Q, P, V> IqlConfig<Q, P, V>
where
    Q: SubModel2<Output = Tensor>,
    P: SubModel1<Output = (Tensor, Tensor)>,
    V: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
{
    /// Sets lambda.
    pub fn lambda(mut self, v: f64) -> Self {
        self.inv_lambda = 1.0 / v;
        self
    }

    /// Sets the numper of parameter update steps per optimization step.
    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    /// Batch size.
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    /// Discount factor.
    pub fn discount_factor(mut self, v: f32) -> Self {
        self.gamma = v;
        self
    }

    // /// Reward scale.
    // ///
    // /// It works for obtaining target values, not the values in logs.
    // pub fn reward_scale(mut self, v: f32) -> Self {
    //     self.reward_scale = v;
    //     self
    // }

    /// Critic loss.
    pub fn critic_loss(mut self, v: CriticLoss) -> Self {
        self.critic_loss = v;
        self
    }

    /// Configuration of value function.
    pub fn value_config(mut self, value_config: ValueConfig<V::Config>) -> Self {
        self.value_config = value_config;
        self
    }

    /// Configuration of actor.
    pub fn actor_config(mut self, actor_config: GaussianActorConfig<P::Config>) -> Self {
        self.actor_config = actor_config;
        self
    }

    /// Configuration of critic.
    pub fn critic_config(mut self, critic_config: MultiCriticConfig<Q::Config>) -> Self {
        self.critic_config = critic_config;
        self
    }

    /// Device.
    pub fn device(mut self, device: candle_core::Device) -> Self {
        self.device = Some(device.into());
        self
    }

    /// If true, advantage weights are calculated with softmax within each mini-batch.
    pub fn adv_softmax(mut self, b: bool) -> Self {
        self.adv_softmax = b;
        self
    }

    /// Saves [`IqlConfig`] to YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ = path.as_ref().to_owned();
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        info!("Save config of IQL agent into {}", path_.to_str().unwrap());
        Ok(())
    }

    /// Constructs [`IqlConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ = path.as_ref().to_owned();
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        info!("Load config of IQL agent from {}", path_.to_str().unwrap());
        Ok(b)
    }
}
