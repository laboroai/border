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
/// Constructs [DQNModel](super::DQNModel).
pub struct DQNModelConfig<Q>
where
    // Q: SubModel<Output = Tensor>,
    // Q::Config: DeserializeOwned + Serialize + OutDim,
    Q: OutDim,
{
    // q_config: Option<Q::Config>,
    pub(super) q_config: Option<Q>,
    pub(super) opt_config: OptimizerConfig,
    // phantom: PhantomData<Q>,
}

// impl<Q: SubModel<Output = Tensor>> Default for DQNModelConfig<Q>
impl<Q> Default for DQNModelConfig<Q>
where
    // Q: SubModel<Output = Tensor>,
    // Q::Config: DeserializeOwned + Serialize + OutDim,
    Q: OutDim,
{
    fn default() -> Self {
        Self {
            q_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
            // phantom: PhantomData,
        }
    }
}

// impl<Q: SubModel<Output = Tensor>> DQNModelConfig<Q>
impl<Q> DQNModelConfig<Q>
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

    /// Constructs [DQNModelConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [DQNModelConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    // /// Constructs [DQNModel] with the given configurations of sub models.
    // pub fn build(self, device: Device) -> Result<DQNModel<Q>> {
    //     let q_config = self.q_config.context("q_config is not set.")?;
    //     let out_dim = q_config.get_out_dim();
    //     let opt_config = self.opt_config;
    //     let var_store = nn::VarStore::new(device);
    //     let q = Q::build(&var_store, q_config);

    //     Ok(DQNModel::_build(
    //         device, out_dim, opt_config, q, var_store, None,
    //     ))
    // }

    // /// Constructs [DQNModel] with the given configurations of sub models.
    // pub fn build_with_submodel_configs(
    //     &self,
    //     q_config: Q::Config,
    //     device: Device,
    // ) -> Result<DQNModel<Q>> {
    //     let out_dim = q_config.get_out_dim();
    //     let opt_config = self.opt_config.clone();
    //     let var_store = nn::VarStore::new(device);
    //     let q = Q::build(&var_store, q_config);

    //     Ok(DQNModel::_build(
    //         device, out_dim, opt_config, q, var_store, None,
    //     ))
    // }
}
