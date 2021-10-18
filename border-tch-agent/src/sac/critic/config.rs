use crate::opt::OptimizerConfig;
use anyhow::Result;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [Critic](super::Critic).
pub struct CriticConfig<Q> {
    pub(super) q_config: Option<Q>,
    pub(super) opt_config: OptimizerConfig,
}

impl<Q> Default for CriticConfig<Q> {
    fn default() -> Self {
        Self {
            q_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
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

    /// Constructs [CriticBuilder] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [CriticBuilder].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
