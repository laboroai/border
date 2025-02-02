//! State value function.
use crate::{
    model::SubModel1,
    opt::{Optimizer, OptimizerConfig},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

/// Configuration of [`Value`].
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct ValueConfig<P> {
    /// Configuration of value function network.
    pub value_config: Option<P>,

    /// Configuration of optimizer.
    pub opt_config: OptimizerConfig,
}

impl<Q> Default for ValueConfig<Q> {
    fn default() -> Self {
        Self {
            value_config: None,
            opt_config: OptimizerConfig::default(),
        }
    }
}

impl<P> ValueConfig<P>
where
    P: DeserializeOwned + Serialize,
{
    /// Sets configurations for value function network.
    pub fn value_config(mut self, v: P) -> Self {
        self.value_config = Some(v);
        self
    }

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Loads [`ValueConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [`ValueConfig`] as YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// State value function.
pub struct Value<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + Clone,
{
    #[allow(dead_code)]
    device: Device, // required when implementing Clone trait
    varmap: VarMap,

    // State-value function
    #[allow(dead_code)]
    value_config: P::Config, // required when implementing Clone trait
    value: P,

    // Optimizer
    #[allow(dead_code)]
    opt_config: OptimizerConfig, // required when implementing Clone trait
    opt: Optimizer,
}

impl<P> Value<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + Clone,
{
    /// Constructs [`Value`].
    pub fn build(config: ValueConfig<P::Config>, device: Device) -> Result<Value<P>> {
        let value_config = config.value_config.context("value_config is not set.")?;
        let varmap = VarMap::new();
        let value = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device).set_prefix("value");
            P::build(vb, value_config.clone())
        };
        let opt_config = config.opt_config;
        let opt = opt_config.build(varmap.all_vars()).unwrap();

        Ok(Self {
            device,
            opt_config,
            varmap,
            opt,
            value,
            value_config,
        })
    }

    /// Returns the state-value for given state (observation).
    pub fn forward(&self, x: &P::Input) -> Tensor {
        self.value.forward(&x)
    }

    /// Backward step for all variables in the value network.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.opt.backward_step(loss)
    }

    /// Save variables to prefix + ".pt".
    pub fn save(&self, prefix: impl AsRef<Path>) -> Result<PathBuf> {
        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("pt");
        self.varmap.save(&path.as_path())?;
        info!("Save value network parameters to {:?}", path);

        Ok(path)
    }

    /// Load variables from prefix + ".pt".
    pub fn load(&mut self, prefix: impl AsRef<Path>) -> Result<()> {
        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("pt");
        self.varmap.load(&path.as_path())?;
        info!("Load value network parameters from {:?}", path);

        Ok(())
    }
}

impl<P> Clone for Value<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + Clone,
{
    fn clone(&self) -> Self {
        unimplemented!();
    }
}
