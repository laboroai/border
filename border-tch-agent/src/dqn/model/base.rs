use super::DqnModelConfig;
use crate::{
    model::{ModelBase, SubModel},
    opt::{Optimizer, OptimizerConfig},
    util::OutDim,
};
use anyhow::Result;
use border_core::record::{
    Record,
    RecordValue::{self, Scalar},
};
use log::{info, trace};
use serde::{de::DeserializeOwned, Serialize};
use std::{marker::PhantomData, path::Path};
use tch::{nn, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
/// Represents value functions for DQN agents.
pub struct DqnModel<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim,
{
    device: Device,
    var_store: nn::VarStore,

    // Dimension of the output vector (equal to the number of actions).
    pub(super) out_dim: i64,

    // Action-value function
    q: Q,

    // Optimizer
    opt_config: OptimizerConfig,
    opt: Optimizer,

    phantom: PhantomData<Q>,
}

impl<Q> DqnModel<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim,
{
    pub fn build(config: DqnModelConfig<Q::Config>, device: Device) -> Self {
        let out_dim = config.q_config.as_ref().unwrap().get_out_dim();
        let opt_config = config.opt_config.clone();
        let var_store = nn::VarStore::new(device);
        let q = Q::build(&var_store, config.q_config.unwrap());

        Self::_build(device, out_dim, opt_config, q, var_store, None)
    }

    fn _build(
        device: Device,
        out_dim: i64,
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
            out_dim,
            opt_config,
            var_store,
            opt,
            q,
            phantom: PhantomData,
        }
    }

    /// Outputs the action-value given an observation.
    pub fn forward(&self, x: &Q::Input) -> Tensor {
        let a = self.q.forward(&x);
        debug_assert_eq!(a.size().as_slice()[1], self.out_dim);
        a
    }

    pub fn param_stats(&self) -> Record {
        crate::util::param_stats(&self.var_store)
    }
}

impl<Q> Clone for DqnModel<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim,
{
    fn clone(&self) -> Self {
        let device = self.device;
        let out_dim = self.out_dim;
        let opt_config = self.opt_config.clone();
        let var_store = nn::VarStore::new(device);
        let q = self.q.clone_with_var_store(&var_store);

        Self::_build(
            device,
            out_dim,
            opt_config,
            q,
            var_store,
            Some(&self.var_store),
        )
    }
}

impl<Q> ModelBase for DqnModel<Q>
where
    Q: SubModel<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim,
{
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn get_var_store(&self) -> &nn::VarStore {
        &self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.var_store.save(&path)?;
        info!("Save DQN model to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.var_store.load(&path)?;
        info!("Load DQN model from {:?}", path.as_ref());
        Ok(())
    }
}
