//! Configuration of SAC agent.
use super::{ActorConfig, CriticConfig};
use crate::{
    model::{SubModel, SubModel2},
    sac::ent_coef::EntCoefMode,
    util::CriticLoss,
    util::OutDim,
    Device,
};
use anyhow::Result;
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};
use tch::Tensor;

/// Constructs [Sac](super::Sac).
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct SacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    pub(super) actor_config: ActorConfig<P::Config>,
    pub(super) critic_config: CriticConfig<Q::Config>,
    pub(super) gamma: f64,
    pub(super) tau: f64,
    pub(super) ent_coef_mode: EntCoefMode,
    pub(super) epsilon: f64,
    pub(super) min_lstd: f64,
    pub(super) max_lstd: f64,
    pub(super) n_updates_per_opt: usize,
    pub(super) min_transitions_warmup: usize,
    pub(super) batch_size: usize,
    pub(super) train: bool,
    pub(super) critic_loss: CriticLoss,
    pub(super) reward_scale: f32,
    pub(super) replay_burffer_capacity: usize,
    pub(super) n_critics: usize,
    pub(super) seed: Option<i64>,
    pub device: Option<Device>,
    // expr_sampling: ExperienceSampling,
}

impl<Q, P> Clone for SacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    fn clone(&self) -> Self {
        Self {
            actor_config: self.actor_config.clone(),
            critic_config: self.critic_config.clone(),
            gamma: self.gamma.clone(),
            tau: self.tau.clone(),
            ent_coef_mode: self.ent_coef_mode.clone(),
            epsilon: self.epsilon.clone(),
            min_lstd: self.min_lstd.clone(),
            max_lstd: self.max_lstd.clone(),
            n_updates_per_opt: self.n_updates_per_opt.clone(),
            min_transitions_warmup: self.min_transitions_warmup.clone(),
            batch_size: self.batch_size.clone(),
            train: self.train.clone(),
            critic_loss: self.critic_loss.clone(),
            reward_scale: self.reward_scale.clone(),
            replay_burffer_capacity: self.replay_burffer_capacity.clone(),
            n_critics: self.n_critics.clone(),
            seed: self.seed.clone(),
            device: self.device.clone(),
        }
    }
}

impl<Q, P> Default for SacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    fn default() -> Self {
        Self {
            actor_config: Default::default(),
            critic_config: Default::default(),
            gamma: 0.99,
            tau: 0.005,
            ent_coef_mode: EntCoefMode::Fix(1.0),
            epsilon: 1e-4,
            min_lstd: -20.0,
            max_lstd: 2.0,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            train: false,
            critic_loss: CriticLoss::MSE,
            reward_scale: 1.0,
            replay_burffer_capacity: 100,
            n_critics: 1,
            seed: None,
            device: None,
            // expr_sampling: ExperienceSampling::Uniform,
        }
    }
}

impl<Q, P> SacConfig<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
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
        self.gamma = v;
        self
    }

    /// Sets soft update coefficient.
    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    /// SAC-alpha.
    pub fn ent_coef_mode(mut self, v: EntCoefMode) -> Self {
        self.ent_coef_mode = v;
        self
    }

    /// Replay buffer capacity.
    pub fn replay_burffer_capacity(mut self, v: usize) -> Self {
        self.replay_burffer_capacity = v;
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
    pub fn device(mut self, device: tch::Device) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Constructs [SacConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ = path.as_ref().to_owned();
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        info!("Load config of SAC agent from {}", path_.to_str().unwrap());
        Ok(b)
    }

    /// Saves [SacConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ = path.as_ref().to_owned();
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        info!("Save config of SAC agent into {}", path_.to_str().unwrap());
        Ok(())
    }
}
