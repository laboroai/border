//! Actor of AWAC agent.
use crate::util::OutDim;
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
    path::Path,
};

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`Actor`].
pub struct ActorConfig<P: OutDim> {
    policy_config: Option<P>,
    opt_config: OptimizerConfig,
}

impl<P: OutDim> Default for ActorConfig<P> {
    fn default() -> Self {
        Self {
            policy_config: None,
            opt_config: OptimizerConfig::default(),
        }
    }
}

impl<P> ActorConfig<P>
where
    P: DeserializeOwned + Serialize + OutDim,
{
    /// Sets configurations for policy function.
    pub fn pi_config(mut self, v: P) -> Self {
        self.policy_config = Some(v);
        self
    }

    /// Sets output dimension of the model.
    pub fn out_dim(mut self, v: i64) -> Self {
        match &mut self.policy_config {
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

    /// Constructs [`ActorConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [`ActorConfig`] as YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// Stochastic policy for AWAC agents.
pub struct Actor<P>
where
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    device: Device,
    varmap: VarMap,

    // Dimension of the action vector.
    out_dim: i64,

    // Action-value function
    policy_config: P::Config,
    policy: P,

    // Optimizer
    opt_config: OptimizerConfig,
    opt: Optimizer,
}

impl<P> Actor<P>
where
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    /// Constructs [`Actor`].
    pub fn build(config: ActorConfig<P::Config>, device: Device) -> Result<Actor<P>> {
        let policy_config = config.policy_config.context("policy_config is not set.")?;
        let out_dim = policy_config.get_out_dim();
        let varmap = VarMap::new();
        let pi = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            P::build(vb, policy_config.clone())
        };
        let opt_config = config.opt_config;

        Ok(Actor::_build(
            device,
            out_dim,
            opt_config,
            policy_config,
            pi,
            varmap,
            None,
        ))
    }

    fn _build(
        device: Device,
        out_dim: i64,
        opt_config: OptimizerConfig,
        policy_config: P::Config,
        policy: P,
        mut varmap: VarMap,
        varmap_src: Option<&VarMap>,
    ) -> Self {
        // Optimizer
        let opt = opt_config.build(varmap.all_vars()).unwrap();

        // Copy varmap
        if let Some(varmap_src) = varmap_src {
            varmap.clone_from(varmap_src);
        }

        Self {
            device,
            out_dim,
            opt_config,
            varmap,
            opt,
            policy,
            policy_config,
        }
    }

    /// Returns the parameters of Gaussian distribution given an observation.
    ///
    /// The type of return values is `(Tensor, Tensor)`.
    /// The shape of the both tensors is `(batch_size, action_dimension)`.
    pub fn forward(&self, x: &P::Input) -> (Tensor, Tensor) {
        let (mean, std) = self.policy.forward(&x);
        debug_assert_eq!(mean.dims()[1], self.out_dim as usize);
        debug_assert_eq!(std.dims()[1], self.out_dim as usize);
        debug_assert_eq!(mean.dims().len(), 2);
        debug_assert_eq!(std.dims().len(), 2);
        (mean, std)
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.opt.backward_step(loss)?;
        Ok(())
    }

    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.varmap.save(&path)?;
        info!("Save actor to {:?}", path.as_ref());
        Ok(())
    }

    pub fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.varmap.load(&path)?;
        info!("Load actor from {:?}", path.as_ref());
        Ok(())
    }
}

impl<P> Clone for Actor<P>
where
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    fn clone(&self) -> Self {
        let device = self.device.clone();
        let opt_config = self.opt_config.clone();
        let varmap = VarMap::new();
        let pi_config = self.policy_config.clone();
        let pi = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            P::build(vb, pi_config.clone())
        };
        let out_dim = self.out_dim;

        Self::_build(
            device,
            out_dim,
            opt_config,
            pi_config,
            pi,
            varmap,
            Some(&self.varmap),
        )
    }
}
