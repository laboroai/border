//! Configuration of behavior cloning (BC) agent.
use super::BcModelConfig;
use crate::{model::SubModel1, util::OutDim, Device};
use anyhow::Result;
use candle_core::Tensor;
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};

/// Action type of behavior cloning (BC) agent.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum BcActionType {
    /// Discrete action.
    Discrete,

    /// Continuous action.
    Continuous,
}

/// Configuration of [`Bc`](super::Bc) agent.
///
/// `P` is the type parameter of the policy model.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct BcConfig<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    pub policy_model_config: BcModelConfig<P::Config>,
    pub batch_size: usize,
    pub action_type: BcActionType,
    pub device: Option<Device>,
    pub record_verbose_level: usize,
    pub phantom: PhantomData<P>,
}

impl<P> Clone for BcConfig<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    fn clone(&self) -> Self {
        Self {
            policy_model_config: self.policy_model_config.clone(),
            batch_size: self.batch_size,
            action_type: self.action_type.clone(),
            device: self.device.clone(),
            record_verbose_level: self.record_verbose_level,
            phantom: PhantomData,
        }
    }
}

impl<P> Default for BcConfig<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    /// Constructs DQN builder with default parameters.
    fn default() -> Self {
        Self {
            policy_model_config: Default::default(),
            batch_size: 1,
            action_type: BcActionType::Discrete,
            device: None,
            record_verbose_level: 0,
            phantom: PhantomData,
        }
    }
}

impl<P> BcConfig<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    /// Sets batch size.
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    /// Sets the configuration of the policy model.
    pub fn policy_model_config(mut self, policy_model_config: BcModelConfig<P::Config>) -> Self {
        self.policy_model_config = policy_model_config;
        self
    }

    /// Sets the output dimention of the agent.
    pub fn out_dim(mut self, out_dim: i64) -> Self {
        let policy_model_config = self.policy_model_config.clone();
        self.policy_model_config = policy_model_config.out_dim(out_dim as _);
        self
    }

    /// Sets device.
    pub fn device(mut self, device: candle_core::Device) -> Self {
        self.device = Some(device.into());
        self
    }

    // Sets action type.
    pub fn action_type(mut self, action_type: BcActionType) -> Self {
        self.action_type = action_type;
        self
    }

    /// Loads [`BcConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ = path.as_ref().to_owned();
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        info!("Load config of BC agent from {}", path_.to_str().unwrap());
        Ok(b)
    }

    /// Saves [`BcConfig`] to YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ = path.as_ref().to_owned();
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        info!("Save config of BC agent into {}", path_.to_str().unwrap());
        Ok(())
    }
}
