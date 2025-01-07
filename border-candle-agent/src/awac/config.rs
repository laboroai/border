//! Configuration of AWAC agent.
use super::{ActorConfig, CriticConfig};
use crate::{
    model::{SubModel1, SubModel2},
    util::CriticLoss,
    util::OutDim,
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

/// Configuration of [`Awac`](super::Awac).
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct AwacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    /// Configuration of the actor model.
    pub actor_config: ActorConfig<P::Config>,

    /// Configuration of the critic model.
    pub critic_config: CriticConfig<Q::Config>,

    /// Discont factor.
    pub gamma: f64,

    /// The inverse of lambda in the paper.
    pub inv_lambda: f64,

    /// Minimum action value
    pub action_min: f32,

    /// Maximum action value
    pub action_max: f32,

    /// Target smoothing coefficient.
    ///
    /// This is a real number between 0 and 1.
    /// A value of 0.001 makes the target network parameters adapt very slowly to the critic network parameters.
    ///
    /// Formula: target_params = tau * critic_params + (1.0 - tau) * target_params
    pub tau: f64,

    /// Minimum value of the log of the standard deviation of the action distribution.
    pub min_lstd: f64,

    /// Maximum value of the log of the standard deviation of the action distribution.
    pub max_lstd: f64,

    /// Number of parameter updates per optimization step.
    pub n_updates_per_opt: usize,

    /// Batch size for training.
    pub batch_size: usize,

    // /// If `true`, the agent is
    // pub train: bool,
    /// Type of critic loss function.
    pub critic_loss: CriticLoss,

    /// Scaling factor for rewards.
    pub reward_scale: f32,

    /// Number of critics used.
    pub n_critics: usize,

    /// Maximum of exponent of advantage.
    pub exp_adv_max: f64,

    /// Random seed value (optional).
    pub seed: Option<i64>,

    /// Device used for the actor and critic models (e.g., CPU or GPU).
    pub device: Option<Device>,
}

impl<Q, P> Clone for AwacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    fn clone(&self) -> Self {
        Self {
            actor_config: self.actor_config.clone(),
            critic_config: self.critic_config.clone(),
            gamma: self.gamma.clone(),
            inv_lambda: self.inv_lambda.clone(),
            tau: self.tau.clone(),
            action_min: self.action_min,
            action_max: self.action_max,
            min_lstd: self.min_lstd.clone(),
            max_lstd: self.max_lstd.clone(),
            n_updates_per_opt: self.n_updates_per_opt.clone(),
            batch_size: self.batch_size.clone(),
            critic_loss: self.critic_loss.clone(),
            reward_scale: self.reward_scale.clone(),
            n_critics: self.n_critics.clone(),
            exp_adv_max: self.exp_adv_max,
            seed: self.seed.clone(),
            device: self.device.clone(),
        }
    }
}

impl<Q, P> Default for AwacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    fn default() -> Self {
        Self {
            actor_config: Default::default(),
            critic_config: Default::default(),
            gamma: 0.99,
            inv_lambda: 10.0,
            tau: 0.005,
            action_min: -1.0,
            action_max: 1.0,
            min_lstd: -20.0,
            max_lstd: 2.0,
            n_updates_per_opt: 1,
            batch_size: 1,
            critic_loss: CriticLoss::Mse,
            reward_scale: 1.0,
            n_critics: 2,
            exp_adv_max: 100.0,
            seed: None,
            device: None,
        }
    }
}

impl<Q, P> AwacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
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
    pub fn discount_factor(mut self, v: f64) -> Self {
        self.gamma = v;
        self
    }

    /// Sets soft update coefficient.
    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    /// Reward scale.
    ///
    /// It works for obtaining target values, not the values in logs.
    pub fn reward_scale(mut self, v: f32) -> Self {
        self.reward_scale = v;
        self
    }

    /// Critic loss.
    pub fn critic_loss(mut self, v: CriticLoss) -> Self {
        self.critic_loss = v;
        self
    }

    /// Configuration of actor.
    pub fn actor_config(mut self, actor_config: ActorConfig<P::Config>) -> Self {
        self.actor_config = actor_config;
        self
    }

    /// Configuration of critic.
    pub fn critic_config(mut self, critic_config: CriticConfig<Q::Config>) -> Self {
        self.critic_config = critic_config;
        self
    }

    /// The number of critics.
    pub fn n_critics(mut self, n_critics: usize) -> Self {
        self.n_critics = n_critics;
        self
    }

    /// Random seed.
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Device.
    pub fn device(mut self, device: candle_core::Device) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Constructs [`AwacConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ = path.as_ref().to_owned();
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        info!("Load config of SAC agent from {}", path_.to_str().unwrap());
        Ok(b)
    }

    /// Saves [`AwacConfig`] to YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ = path.as_ref().to_owned();
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        info!("Save config of SAC agent into {}", path_.to_str().unwrap());
        Ok(())
    }
}
