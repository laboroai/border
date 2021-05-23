//! Actor of SAC agent.
use super::super::{
    model::{ModelBase, SubModel},
    opt::{Optimizer, OptimizerConfig},
    util::OutDim,
};
use anyhow::{Context, Result};
use log::{info, trace};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};
use tch::{nn, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Constructs [Actor].
pub struct ActorBuilder<P: SubModel<Output = (Tensor, Tensor)>>
where
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    pi_config: Option<P::Config>,
    opt_config: OptimizerConfig,
}

impl<P: SubModel<Output = (Tensor, Tensor)>> Default for ActorBuilder<P>
where
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    fn default() -> Self {
        Self {
            pi_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
            // phantom: PhantomData,
        }
    }
}

impl<P: SubModel<Output = (Tensor, Tensor)>> ActorBuilder<P>
where
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    /// Sets configurations for action-value function.
    pub fn pi_config(mut self, v: P::Config) -> Self {
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

    /// Constructs [ActorBuilder] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [ActorBuilder].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    /// Constructs [Actor] with the given configurations of sub models.
    pub fn build(self, device: Device) -> Result<Actor<P>> {
        let pi_config = self.pi_config.context("pi_config is not set.")?;
        let out_dim = pi_config.get_out_dim();
        let opt_config = self.opt_config;
        let var_store = nn::VarStore::new(device);
        let pi = P::build(&var_store, pi_config);

        Ok(Actor::_build(
            device, out_dim, opt_config, pi, var_store, None,
        ))
    }

    /// Constructs [Actor] with the given configurations of sub models.
    pub fn build_with_submodel_configs(
        &self,
        pi_config: P::Config,
        device: Device,
    ) -> Result<Actor<P>> {
        let out_dim = pi_config.get_out_dim();
        let opt_config = self.opt_config.clone();
        let var_store = nn::VarStore::new(device);
        let pi = P::build(&var_store, pi_config);

        Ok(Actor::_build(
            device, out_dim, opt_config, pi, var_store, None,
        ))
    }
}

#[allow(clippy::upper_case_acronyms)]
/// Represents a stochastic policy for SAC agents.
pub struct Actor<P>
where
    P: SubModel<Output = (Tensor, Tensor)>,
{
    device: Device,
    var_store: nn::VarStore,

    // Dimension of the output vector (equal to the number of actions).
    pub(super) out_dim: i64,

    // Action-value function
    pi: P,

    // Optimizer
    opt_config: OptimizerConfig,
    opt: Optimizer,
}

impl<P> Actor<P>
where
    P: SubModel<Output = (Tensor, Tensor)>,
{
    fn _build(
        device: Device,
        out_dim: i64,
        opt_config: OptimizerConfig,
        pi: P,
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
            out_dim,
            opt_config,
            var_store,
            opt,
            pi,
        }
    }

    /// Outputs the parameters of Gaussian distribution given an observation.
    pub fn forward(&self, x: &P::Input) -> (Tensor, Tensor) {
        let (mean, std) = self.pi.forward(&x);
        debug_assert_eq!(mean.size().as_slice()[1], self.out_dim);
        debug_assert_eq!(std.size().as_slice()[1], self.out_dim);
        (mean, std)
    }
}

impl<P> Clone for Actor<P>
where
    P: SubModel<Output = (Tensor, Tensor)>,
{
    fn clone(&self) -> Self {
        let device = self.device;
        let out_dim = self.out_dim;
        let opt_config = self.opt_config.clone();
        let var_store = nn::VarStore::new(device);
        let pi = self.pi.clone_with_var_store(&var_store);

        Self::_build(
            device,
            out_dim,
            opt_config,
            pi,
            var_store,
            Some(&self.var_store),
        )
    }
}

impl<P> ModelBase for Actor<P>
where
    P: SubModel<Output = (Tensor, Tensor)>,
{
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save actor to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load actor from {:?}", path.as_ref());
        Ok(())
    }
}
