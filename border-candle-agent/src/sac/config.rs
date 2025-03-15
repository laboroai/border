//! Configuration of SAC agent.
use crate::{
    model::{SubModel1, SubModel2},
    sac2::ent_coef::EntCoefMode,
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

/// Configuration of [`Sac2`](super::Sac2).
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Sac2Config<Q, P>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Debug + PartialEq + Clone,
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Debug + PartialEq + Clone,
{
    /// Configuration of the actor model.
    pub actor_config: GaussianActorConfig<P::Config>,

    /// Configuration of the critic model.
    pub critic_config: MultiCriticConfig<Q::Config>,

    /// Discont factor.
    pub gamma: f64,

    /// How to update entropy coefficient.
    pub ent_coef_mode: EntCoefMode,

    /// Number of parameter updates per optimization step.
    pub n_updates_per_opt: usize,

    /// Batch size for training.
    pub batch_size: usize,

    /// Type of critic loss function.
    pub critic_loss: CriticLoss,

    /// Device for actor/critic models.
    pub device: Option<Device>,
}

impl<Q, P> Clone for Sac2Config<Q, P>
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
            ent_coef_mode: self.ent_coef_mode.clone(),
            n_updates_per_opt: self.n_updates_per_opt.clone(),
            batch_size: self.batch_size.clone(),
            critic_loss: self.critic_loss.clone(),
            device: self.device.clone(),
        }
    }
}

impl<Q, P> Default for Sac2Config<Q, P>
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
            ent_coef_mode: EntCoefMode::Fix(1.0),
            n_updates_per_opt: 1,
            batch_size: 1,
            critic_loss: CriticLoss::Mse,
            device: None,
        }
    }
}

impl<Q, P> Sac2Config<Q, P>
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

    /// SAC-alpha.
    pub fn ent_coef_mode(mut self, v: EntCoefMode) -> Self {
        self.ent_coef_mode = v;
        self
    }

    /// Critic loss.
    pub fn critic_loss(mut self, v: CriticLoss) -> Self {
        self.critic_loss = v;
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

    /// Constructs [`Sac2Config`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ = path.as_ref().to_owned();
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        info!("Load config of SAC agent from {}", path_.to_str().unwrap());
        Ok(b)
    }

    /// Saves [`Sac2Config`].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ = path.as_ref().to_owned();
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        info!("Save config of SAC agent into {}", path_.to_str().unwrap());
        Ok(())
    }
}
