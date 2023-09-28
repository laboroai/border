//! Critic of SAC agent.
use crate::{opt::{Optimizer, OptimizerConfig}, model::SubModel2};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};
use log::info;

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`Critic`].
pub struct CriticConfig<Q> {
    pub q_config: Option<Q>,
    pub opt_config: OptimizerConfig,
}

impl<Q> Default for CriticConfig<Q> {
    fn default() -> Self {
        Self {
            q_config: None,
            opt_config: OptimizerConfig::default(),
        }
    }
}

impl<Q> CriticConfig<Q>
where
    Q: DeserializeOwned + Serialize,
{
    /// Sets configurations for action-value function.
    pub fn q_config(mut self, v: Q) -> Self {
        self.q_config = Some(v);
        self
    }

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Constructs [CriticConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [CriticConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// Represents soft critic for SAC agents.
///
/// It takes observations and actions as inputs and outputs action values.
pub struct Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize,
{
    device: Device,
    varmap: VarMap,

    /// Action-value function
    q: Q,
    q_config: Q::Config,

    opt_config: OptimizerConfig,
    opt: Optimizer,
}

impl<Q> Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Clone,
{
    /// Constructs [`Critic`].
    pub fn build(config: CriticConfig<Q::Config>, device: Device) -> Result<Critic<Q>> {
        let q_config = config.q_config.context("q_config is not set.")?;
        let opt_config = config.opt_config;
        let varmap = VarMap::new();
        let q = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            Q::build(vb, q_config.clone())
        };

        Ok(Critic::_build(
            device, opt_config, q_config, q, varmap, None,
        ))
    }

    fn _build(
        device: Device,
        opt_config: OptimizerConfig,
        q_config: Q::Config,
        q: Q,
        mut varmap: VarMap,
        varmap_src: Option<&VarMap>,
    ) -> Self {
        // Optimizer
        let opt = opt_config.build(varmap.all_vars()).unwrap();

        // Copy varmap
        if let Some(varmap_src) = varmap_src {
            varmap.clone_from(varmap_src);
        }

        Self {
            device,
            opt_config,
            varmap,
            opt,
            q,
            q_config,
        }
    }

    /// Outputs the action-value given observations and actions.
    pub fn forward(&self, obs: &Q::Input1, act: &Q::Input2) -> Tensor {
        self.q.forward(obs, act)
    }
}

impl<Q> Clone for Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Clone,
{
    fn clone(&self) -> Self {
        let device = self.device.clone();
        let opt_config = self.opt_config.clone();
        let varmap = VarMap::new();
        let q_config = self.q_config.clone();
        let q = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            Q::build(vb, q_config.clone())
        };

        Self::_build(device, opt_config, q_config, q, varmap, Some(&self.varmap))
    }
}

impl<Q> Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize,
{
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.opt.backward_step(loss)
    }

    // fn get_var_store_mut(&mut self) -> &mut nn::VarStore {
    //     &mut self.var_store
    // }

    pub fn get_varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.varmap.save(&path)?;
        info!("Save critic to {:?}", path.as_ref());
        Ok(())
    }

    pub fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.varmap.load(&path)?;
        info!("Load critic from {:?}", path.as_ref());
        Ok(())
    }
}
