use crate::{
    model::SubModel1,
    opt::{Optimizer, OptimizerConfig},
    util::OutDim,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`DqnModel`].
pub struct DqnModelConfig<Q>
where
    Q: OutDim,
{
    pub(super) q_config: Option<Q>,
    pub(super) opt_config: OptimizerConfig,
}

impl<Q> Default for DqnModelConfig<Q>
where
    Q: OutDim,
{
    fn default() -> Self {
        Self {
            q_config: None,
            opt_config: OptimizerConfig::default(),
        }
    }
}

impl<Q> DqnModelConfig<Q>
where
    Q: DeserializeOwned + Serialize + OutDim,
{
    /// Sets configurations for action-value function.
    pub fn q_config(mut self, v: Q) -> Self {
        self.q_config = Some(v);
        self
    }

    /// Sets output dimension of the model.
    pub fn out_dim(mut self, v: i64) -> Self {
        match &mut self.q_config {
            None => {}
            Some(q_config) => q_config.set_out_dim(v),
        };
        self
    }

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Constructs [`DqnModelConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [`DqnModelConfig`] to as a YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

pub struct DqnModel<Q>
where
    Q: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim,
{
    device: Device,
    varmap: VarMap,

    // Dimension of the output vector (equal to the number of actions).
    pub(super) out_dim: i64,

    // Action-value function
    q: Q,

    // Optimizer
    opt_config: OptimizerConfig,
    q_config: Q::Config,
    opt: Optimizer,
}

impl<Q> DqnModel<Q>
where
    Q: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    /// Constructs [`DqnModel`].
    pub fn build(config: DqnModelConfig<Q::Config>, device: Device) -> Result<Self> {
        let out_dim = config.q_config.as_ref().unwrap().get_out_dim();
        let q_config = config.q_config.context("q_config is not set.")?;
        let opt_config = config.opt_config;
        let varmap = VarMap::new();
        let q = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            Q::build(vb, q_config.clone())
        };

        Ok(Self::_build(
            device,
            out_dim as _,
            opt_config,
            q_config,
            q,
            varmap,
            None,
        ))
    }

    fn _build(
        device: Device,
        out_dim: i64,
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
            out_dim,
            opt_config,
            varmap,
            opt,
            q,
            q_config,
        }
    }

    /// Outputs the action-value given observation(s).
    pub fn forward(&self, obs: &Q::Input) -> Tensor {
        self.q.forward(obs)
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.opt.backward_step(loss)
    }

    pub fn get_varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.varmap.save(&path)?;
        info!("Save dqnmodel to {:?}", path.as_ref());
        Ok(())
    }

    pub fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.varmap.load(&path)?;
        info!("Load dqnmodel from {:?}", path.as_ref());
        Ok(())
    }
}

impl<Q> Clone for DqnModel<Q>
where
    Q: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    fn clone(&self) -> Self {
        let device = self.device.clone();
        let out_dim = self.out_dim;
        let opt_config = self.opt_config.clone();
        let q_config = self.q_config.clone();
        let varmap = VarMap::new();
        let q = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            Q::build(vb, self.q_config.clone())
        };

        Self::_build(
            device,
            out_dim,
            opt_config,
            q_config,
            q,
            varmap,
            Some(&self.varmap),
        )
    }
}
