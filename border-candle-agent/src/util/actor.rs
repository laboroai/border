//! Actor with Gaussian policy.
use crate::{
    model::SubModel1,
    opt::{Optimizer, OptimizerConfig},
    util::{atanh, log_jacobian_tanh, OutDim},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

fn normal_logp(x: &Tensor) -> Result<Tensor> {
    let tmp: Tensor =
        ((-0.5 * (2.0 * std::f32::consts::PI).ln() as f64) - (0.5 * x.powf(2.0)?)?)?;
    Ok(tmp.sum(D::Minus1)?)
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`GaussianActor`].
pub struct GaussianActorConfig<P: OutDim> {
    pub policy_config: Option<P>,
    pub opt_config: OptimizerConfig,
    pub min_log_std: f64,
    pub max_log_std: f64,
}

impl<P: OutDim> Default for GaussianActorConfig<P> {
    fn default() -> Self {
        Self {
            policy_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0003 },
            min_log_std: -20.0,
            max_log_std: 2.0,
        }
    }
}

impl<P> GaussianActorConfig<P>
where
    P: DeserializeOwned + Serialize + OutDim,
{
    /// Sets the minimum value of log std.
    pub fn min_log_std(mut self, v: f64) -> Self {
        self.min_log_std = v;
        self
    }

    /// Sets the maximum value of log std.
    pub fn max_log_std(mut self, v: f64) -> Self {
        self.max_log_std = v;
        self
    }

    /// Sets configurations for policy function.
    pub fn policy_config(mut self, v: P) -> Self {
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

    /// Loads [`ActorConfig`] from YAML file.
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

/// Actor with Gaussian policy.
pub struct GaussianActor<P>
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

    // Min/max log std
    min_log_std: f64,
    max_log_std: f64,
}

impl<P> GaussianActor<P>
where
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    /// Constructs [`Actor`].
    pub fn build(
        config: GaussianActorConfig<P::Config>,
        device: Device,
    ) -> Result<GaussianActor<P>> {
        let min_log_std = config.min_log_std;
        let max_log_std = config.max_log_std;
        let policy_config = config.policy_config.context("policy_config is not set.")?;
        let out_dim = policy_config.get_out_dim();
        let varmap = VarMap::new();
        let policy = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device).set_prefix("actor");
            P::build(vb, policy_config.clone())
        };
        let opt_config = config.opt_config;
        let opt = opt_config.build(varmap.all_vars()).unwrap();

        Ok(Self {
            device,
            out_dim,
            opt_config,
            varmap,
            opt,
            policy,
            policy_config,
            min_log_std,
            max_log_std,
        })
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

    /// Rerurns the log probabilities (densities) of the given actions
    pub fn logp<'a>(&self, obs: &P::Input, act: &Tensor) -> Result<Tensor> {
        // Distribution parameters on the given observation
        log::trace!("Distribution parameters on the given observation");
        let (mean, std) = {
            let (mean, lstd) = self.forward(obs);
            let std = lstd.clamp(self.min_log_std, self.max_log_std)?.exp()?;
            (mean, std)
        };

        // Inverse transformation to the standard normal: N(0, 1)
        log::trace!("Inverse transformation to the standard normal: N(0, 1)");
        let act = atanh(&act.to_device(&self.device)?)?;
        let z = ((&act - &mean)? / &std)?;

        // Density
        log::trace!("Density");
        Ok((normal_logp(&z)? + log_jacobian_tanh(&act, 1e-4, &self.device)?)?)
    }

    /// Samples actions.
    ///
    /// If `train` is `true`, actions are sampled from a Gaussian distribution.
    /// Otherwise, the mean of the Gaussian distribution is returned.
    pub fn sample(&mut self, obs: &P::Input, train: bool) -> Tensor {
        let (mean, lstd) = self.forward(&obs);
        let std = lstd
            .clamp(self.min_log_std, self.max_log_std)
            .unwrap()
            .exp()
            .unwrap();
        match train {
            true => ((std * mean.randn_like(0., 1.).unwrap()).unwrap() + mean)
                .unwrap()
                .tanh()
                .unwrap(),
            false => mean.tanh().unwrap(),
        }
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.opt.backward_step(loss)?;
        Ok(())
    }

    /// Save variables to prefix + ".pt".
    pub fn save(&self, prefix: impl AsRef<Path>) -> Result<PathBuf> {
        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("pt");
        self.varmap.save(&path.as_path())?;
        info!("Save actor parameters to {:?}", path);

        Ok(path.to_path_buf())
    }

    /// Load variables from prefix + ".pt".
    pub fn load(&mut self, prefix: impl AsRef<Path>) -> Result<()> {
        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("pt");
        self.varmap.load(&path.as_path())?;
        info!("Load actor parameters from {:?}", path);

        Ok(())
    }
}

impl<P> Clone for GaussianActor<P>
where
    P: SubModel1<Output = (Tensor, Tensor)>,
    P::Config: DeserializeOwned + Serialize + OutDim + Clone,
{
    fn clone(&self) -> Self {
        let min_log_std = self.min_log_std;
        let max_log_std = self.max_log_std;
        let device = self.device.clone();
        let opt_config = self.opt_config.clone();
        let mut varmap = VarMap::new();
        let policy_config = self.policy_config.clone();
        let policy = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            P::build(vb, policy_config.clone())
        };
        let out_dim = self.out_dim;
        let opt = opt_config.build(varmap.all_vars()).unwrap();

        // Copy varmap
        varmap.clone_from(&self.varmap);

        Self {
            device,
            out_dim,
            opt_config,
            varmap,
            opt,
            policy,
            policy_config,
            min_log_std,
            max_log_std,
        }
    }
}
