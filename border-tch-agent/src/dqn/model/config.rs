use crate::{opt::OptimizerConfig, util::OutDim};
use anyhow::Result;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [DqnModel](super::DqnModel).
pub struct DqnModelConfig<Q>
where
    // Q: SubModel<Output = Tensor>,
    // Q::Config: DeserializeOwned + Serialize + OutDim,
    Q: OutDim,
{
    pub q_config: Option<Q>,
    pub opt_config: OptimizerConfig,
}

// impl<Q: SubModel<Output = Tensor>> Default for DQNModelConfig<Q>
impl<Q> Default for DqnModelConfig<Q>
where
    // Q: SubModel<Output = Tensor>,
    // Q::Config: DeserializeOwned + Serialize + OutDim,
    Q: OutDim,
{
    fn default() -> Self {
        Self {
            q_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
        }
    }
}

// impl<Q: SubModel<Output = Tensor>> DQNModelConfig<Q>
impl<Q> DqnModelConfig<Q>
where
    // Q: SubModel<Output = Tensor>,
    // Q::Config: DeserializeOwned + Serialize + OutDim,
    Q: DeserializeOwned + Serialize + OutDim,
{
    /// Sets configurations for action-value function.
    // pub fn q_config(mut self, v: Q::Config) -> Self {
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

    /// Constructs [DqnModelConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [DqnModelConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
