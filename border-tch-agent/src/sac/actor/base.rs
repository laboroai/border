use super::ActorConfig;
use crate::{
    model::{ModelBase, SubModel},
    opt::{Optimizer, OptimizerConfig},
    util::OutDim,
};
use anyhow::{Context, Result};
use log::{info, trace};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    path::Path,
};
use tch::{nn, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
/// Represents a stochastic policy for SAC agents.
pub struct Actor<P>
where
    P: SubModel<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    device: Device,
    var_store: nn::VarStore,

    // Dimension of the action vector.
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
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    /// Constructs [Actor].
    pub fn build(config: ActorConfig<P::Config>, device: Device) -> Result<Actor<P>> {
        let pi_config = config.pi_config.context("pi_config is not set.")?;
        let out_dim = pi_config.get_out_dim();
        let opt_config = config.opt_config;
        let var_store = nn::VarStore::new(device);
        let pi = P::build(&var_store, pi_config);

        Ok(Actor::_build(
            device, out_dim, opt_config, pi, var_store, None,
        ))
    }

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
    P::Config: DeserializeOwned + Serialize + OutDim,
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
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.var_store.save(&path)?;
        info!("Save actor to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.var_store.load(&path)?;
        info!("Load actor from {:?}", path.as_ref());
        Ok(())
    }
}
