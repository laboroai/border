use crate::{opt::OptimizerConfig, util::OutDim};
use anyhow::Result;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`Actor`](super::Actor).
pub struct ActorConfig<P: OutDim> {
    pub pi_config: Option<P>,
    pub opt_config: OptimizerConfig,
}

impl<P: OutDim> Default for ActorConfig<P> {
    fn default() -> Self {
        Self {
            pi_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
        }
    }
}

impl<P> ActorConfig<P>
where
    P: DeserializeOwned + Serialize + OutDim,
{
    /// Sets configurations for action-value function.
    pub fn pi_config(mut self, v: P) -> Self {
        self.pi_config = Some(v);
        self
    }

    /// Sets output dimension of the model.
    pub fn out_dim(mut self, v: i64) -> Self {
        match &mut self.pi_config {
            None => {}
            Some(pi_config) => pi_config.set_out_dim(v),
        };
        self
    }

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Constructs [ActorConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [ActorConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
