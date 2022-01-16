//! IQN model.
use crate::{
    model::SubModel,
    opt::OptimizerConfig,
    util::OutDim,
};
use anyhow::Result;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    default::Default,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[cfg(not(feature = "adam_eps"))]
impl<F: SubModel, M: SubModel> IqnModelConfig<F, M>
where
    F::Config: DeserializeOwned + Serialize,
    M::Config: DeserializeOwned + Serialize,
{
    /// Sets the learning rate.
    pub fn learning_rate(mut self, v: f64) -> Self {
        match &self.opt_config {
            OptimizerConfig::Adam { lr: _ } => self.opt_config = OptimizerConfig::Adam { lr: v },
        };
        self
    }
}

// #[cfg(feature = "adam_eps")]
// impl<F: SubModel, M: SubModel> IqnModelConfig<F, M>
// where
//     F::Config: DeserializeOwned + Serialize,
//     M::Config: DeserializeOwned + Serialize,
// {
//     /// Sets the learning rate.
//     pub fn learning_rate(mut self, v: f64) -> Self {
//         match &self.opt_config {
//             OptimizerConfig::Adam { lr: _ } => self.opt_config = OptimizerConfig::Adam { lr: v },
//             _ => unimplemented!(),
//         };
//         self
//     }
// }

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [IqnModel].
///
/// The type parameter `F` represents a configuration struct of a feature extractor.
/// The type parameter `M` represents a configuration struct of a model for merging
/// cosine-embedded percent points and feature vectors.
pub struct IqnModelConfig<F, M>
where
    F: DeserializeOwned + Serialize,
    M: DeserializeOwned + Serialize,
{
    /// Dimension of feature vector.
    pub feature_dim: i64,

    /// Embedding dimension.
    pub embed_dim: i64,

    /// Configuration of feature extractor.
    pub f_config: Option<F>,

    /// Configuration of a model for merging percentils and feature vectors.
    pub m_config: Option<M>,

    /// Configuration of optimizer.
    pub opt_config: OptimizerConfig,
}

impl<F, M> Default for IqnModelConfig<F, M>
where
    F: DeserializeOwned + Serialize,
    M: DeserializeOwned + Serialize,
{
    fn default() -> Self {
        Self {
            feature_dim: 0,
            embed_dim: 0,
            f_config: None,
            m_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
        }
    }
}

impl<F, M> IqnModelConfig<F, M>
where
    F: DeserializeOwned + Serialize,
    M: DeserializeOwned + Serialize,
{
    /// Sets the dimension of cos-embedding of percent points.
    pub fn embed_dim(mut self, v: i64) -> Self {
        self.embed_dim = v;
        self
    }

    /// Sets the dimension of feature vectors.
    pub fn feature_dim(mut self, v: i64) -> Self {
        self.feature_dim = v;
        self
    }

    /// Sets configurations for feature extractor.
    pub fn f_config(mut self, v: F::Config) -> Self {
        self.f_config = Some(v);
        self
    }

    /// Sets configurations for output model.
    pub fn m_config(mut self, v: M::Config) -> Self {
        self.m_config = Some(v);
        self
    }

    /// Sets output dimension of the model.
    pub fn out_dim(mut self, v: i64) -> Self {
        match &mut self.m_config {
            None => {}
            Some(m_config) => m_config.set_out_dim(v),
        };
        self
    }

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Constructs [IQNModelBuilder] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [IQNModelBuilder].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
