//! Critic of SAC agent.
use super::super::{
    model::{ModelBase, SubModel2},
    opt::{Optimizer, OptimizerConfig},
};
use anyhow::{Context, Result};
use log::{info, trace};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};
use tch::{nn, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Constructs [Critic].
pub struct CriticBuilder<Q: SubModel2<Output = Tensor>>
where
    Q::Config: DeserializeOwned + Serialize,
{
    q_config: Option<Q::Config>,
    opt_config: OptimizerConfig,
}

impl<Q: SubModel2<Output = Tensor>> Default for CriticBuilder<Q>
where
    Q::Config: DeserializeOwned + Serialize,
{
    fn default() -> Self {
        Self {
            q_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
        }
    }
}

impl<Q: SubModel2<Output = Tensor>> CriticBuilder<Q>
where
    Q::Config: DeserializeOwned + Serialize,
{
    /// Sets configurations for action-value function.
    pub fn q_config(mut self, v: Q::Config) -> Self {
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

    /// Constructs [Critic] with the given configurations of sub models.
    pub fn build(self, device: Device) -> Result<Critic<Q>> {
        let q_config = self.q_config.context("q_config is not set.")?;
        let opt_config = self.opt_config;
        let var_store = nn::VarStore::new(device);
        let q = Q::build(&var_store, q_config);

        Ok(Critic::_build(
            device, opt_config, q, var_store, None,
        ))
    }

    /// Constructs [IQNModel] with the given configurations of sub models.
    pub fn build_with_submodel_configs(
        &self,
        q_config: Q::Config,
        device: Device,
    ) -> Result<Critic<Q>> {
        let opt_config = self.opt_config.clone();
        let var_store = nn::VarStore::new(device);
        let q = Q::build(&var_store, q_config);

        Ok(Critic::_build(
            device, opt_config, q, var_store, None,
        ))
    }
}


#[allow(clippy::upper_case_acronyms)]
/// Represents soft critic for SAC agents.
///
/// It takes observations and actions as inputs and outputs action values.
pub struct Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
{
    device: Device,
    var_store: nn::VarStore,

    // Action-value function
    q: Q,

    // Optimizer
    opt_config: OptimizerConfig,
    opt: Optimizer,
}

impl<Q> Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
{
    fn _build(
        device: Device,
        opt_config: OptimizerConfig,
        q: Q,
        mut var_store: nn::VarStore,
        var_store_src: Option<&nn::VarStore>,
    ) -> Self {
        // Optimizer
        let opt = opt_config.build(&var_store).unwrap();

        // Copy var_store
        if let Some(var_store_src) = var_store_src {
            var_store.copy(var_store_src).unwrap();
        }

        Self {
            device,
            opt_config,
            var_store,
            opt,
            q,
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
{
    fn clone(&self) -> Self {
        let device = self.device;
        let opt_config = self.opt_config.clone();
        let var_store = nn::VarStore::new(device);
        let q = self.q.clone_with_var_store(&var_store);
    
        Self::_build(
            device,
            opt_config,
            q,
            var_store,
            Some(&self.var_store),
        )    
    }
}

impl<Q> ModelBase for Critic<Q>
where
    Q: SubModel2<Output = Tensor>,
{
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save critic to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load critic from {:?}", path.as_ref());
        Ok(())
    }
}