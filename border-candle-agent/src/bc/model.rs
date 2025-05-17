use crate::{
    model::SubModel1,
    opt::{Optimizer, OptimizerConfig},
    util::OutDim,
};
use anyhow::{Context, Result};
use border_core::record::Record;
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
/// Configuration of [`BcModel`].
///
/// The type parameter `C` should be a configuration of policy model, which should outputs a tensor.
/// The policy model supports both discrete and continuous action spaces, leaving the interpretation
/// of the output to the caller.
pub struct BcModelConfig<C>
where
    C: OutDim + Clone,
{
    pub policy_model_config: Option<C>,
    #[serde(default)]
    pub opt_config: OptimizerConfig,
}

impl<C> Default for BcModelConfig<C>
where
    C: DeserializeOwned + Serialize + OutDim + Clone,
{
    fn default() -> Self {
        Self {
            policy_model_config: None,
            opt_config: OptimizerConfig::default(),
        }
    }
}

impl<C> BcModelConfig<C>
where
    C: DeserializeOwned + Serialize + OutDim + Clone,
{
    /// Sets configurations for the policy model.
    pub fn policy_model_config(mut self, v: C) -> Self {
        self.policy_model_config = Some(v);
        self
    }

    /// Sets output dimension of the model.
    pub fn out_dim(mut self, v: i64) -> Self {
        match &mut self.policy_model_config {
            None => {}
            Some(policy_model_config) => policy_model_config.set_out_dim(v),
        };
        self
    }

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Constructs [`BcModelConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [`BcModelConfig`] to as a YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// Policy model for behaviour cloning.
///
/// The model's architecture is specified by the type parameter `P`,
/// which must implement [`SubModel1`]. It takes [`SubModel1::Input`] as
/// input and produces a tensor as output.
pub struct BcModel<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim,
{
    device: Device,
    varmap: VarMap,

    /// Dimension of the output vector.
    out_dim: i64,

    /// Policy model.
    policy_model: P,

    /// Optimizer configuration.
    opt_config: OptimizerConfig,

    /// Optimizer.
    opt: Optimizer,

    /// Policy model configuration.
    policy_model_config: P::Config,
}

impl<P> BcModel<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    /// Constructs [`BcModel`].
    pub fn build(config: BcModelConfig<P::Config>, device: Device) -> Result<Self> {
        let out_dim = config.policy_model_config.as_ref().unwrap().get_out_dim();
        let policy_model_config = config
            .policy_model_config
            .context("policy_model_config is not set.")?;
        let opt_config = config.opt_config;
        let varmap = VarMap::new();

        // Build policy model
        let policy_model = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            P::build(vb, policy_model_config.clone())
        };

        Ok(Self::_build(
            device,
            out_dim as _,
            opt_config,
            policy_model_config,
            policy_model,
            varmap,
            None,
        ))
    }

    fn _build(
        device: Device,
        out_dim: i64,
        opt_config: OptimizerConfig,
        policy_model_config: P::Config,
        policy_model: P,
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
            policy_model,
            policy_model_config,
        }
    }

    /// Outputs the action-value given observation(s).
    pub fn forward(&self, obs: &P::Input) -> Tensor {
        self.policy_model.forward(obs)
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        // Consider to use gradient clipping, below code
        // let mut grads = loss.backward()?;
        // for (_, var) in self.varmap.data().lock().unwrap().iter() {
        //     let g1 = grads.get(var).unwrap();
        //     let g2 = g1.clamp(-1.0, 1.0)?;
        //     let _ = grads.remove(&var).unwrap();
        //     let _ = grads.insert(&var, g2);
        // }
        // self.opt.step(&grads)
        self.opt.backward_step(loss)
    }

    pub fn get_varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.varmap.save(&path)?;
        info!("Save bc model to {:?}", path.as_ref());
        Ok(())
    }

    pub fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.varmap.load(&path)?;
        info!("Load bc model from {:?}", path.as_ref());
        Ok(())
    }

    pub fn param_stats(&self) -> Record {
        crate::util::param_stats(&self.varmap)
    }
}

impl<P> Clone for BcModel<P>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    fn clone(&self) -> Self {
        let device = self.device.clone();
        let out_dim = self.out_dim;
        let opt_config = self.opt_config.clone();
        let policy_model_config = self.policy_model_config.clone();
        let varmap = VarMap::new();
        let policy_model = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            P::build(vb, self.policy_model_config.clone())
        };

        Self::_build(
            device,
            out_dim,
            opt_config,
            policy_model_config,
            policy_model,
            varmap,
            Some(&self.varmap),
        )
    }
}
