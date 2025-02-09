//! Critic for agents with continuous action.
use crate::{
    model::SubModel2,
    opt::{Optimizer, OptimizerConfig},
    util::track_with_replace_substring,
};
use anyhow::{Context, Result};
use candle_core::{DType::F32, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`MultiCritic`].
pub struct MultiCriticConfig<Q> {
    /// The number of critic networks.
    pub n_nets: usize,

    /// Configuration of critic networks.
    pub q_config: Option<Q>,

    /// Configuration of the optimizer.
    pub opt_config: OptimizerConfig,

    /// Soft update coefficient.
    pub tau: f64,
}

impl<Q> Default for MultiCriticConfig<Q> {
    fn default() -> Self {
        Self {
            n_nets: 2,
            q_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0003 },
            tau: 0.005,
        }
    }
}

impl<Q> MultiCriticConfig<Q>
where
    Q: DeserializeOwned + Serialize,
{
    /// Sets the numver of critic networks.
    pub fn n_nets(mut self, v: usize) -> Self {
        self.n_nets = v;
        self
    }

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

    /// Sets soft update parameter tau.
    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    /// Constructs [`MultiCriticConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [`MultiCriticConfig`] as YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// Critic for agents with continuous action.
///
/// It takes observations and actions as inputs and outputs action values.
///
/// This struct has multiple q functions and corresponding target networks.
pub struct MultiCritic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize,
{
    n_nets: usize,
    tau: f64,
    device: Device,
    varmap: VarMap,
    varmap_tgt: VarMap, // for target network

    /// Action-value function
    q_config: Q::Config,
    qs: Vec<Q>,
    qs_tgt: Vec<Q>, // for target network

    opt_config: OptimizerConfig,
    opt: Optimizer, // no optimizer required for tatget networks
}

impl<Q> MultiCritic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Clone,
{
    /// Constructs [`MultiCritic`].
    pub fn build(config: MultiCriticConfig<Q::Config>, device: Device) -> Result<MultiCritic<Q>> {
        let tau = config.tau;
        let n_nets = config.n_nets;
        let q_config = config.q_config.context("q_config is not set.")?;
        let opt_config = config.opt_config;

        // Critic networks
        let (varmap, qs) = Self::build_critic_networks(&q_config, &device, n_nets, "critic");

        // Target networks
        let (varmap_tgt, qs_tgt) =
            Self::build_critic_networks(&q_config, &device, n_nets, "critic_tgt");

        // Optimizer, shared with critic networks
        let opt = opt_config.build(varmap.all_vars())?;

        // Copy parameters
        track_with_replace_substring(&varmap_tgt, &varmap, 1.0, ("critic", "critic_tgt"))?;

        Ok(Self {
            tau,
            n_nets,
            device,
            varmap,
            varmap_tgt,
            q_config,
            qs,
            qs_tgt,
            opt_config,
            opt,
        })
    }

    fn build_critic_networks(
        q_config: &Q::Config,
        device: &Device,
        n_nets: usize,
        prefix: &str,
    ) -> (VarMap, Vec<Q>) {
        let varmap = VarMap::new();
        let qs = (0..n_nets)
            .map(|ix| {
                if device.is_cuda() {
                    device.set_seed((ix + 10) as _).unwrap();
                }
                let vb = VarBuilder::from_varmap(&varmap, F32, &device)
                    .set_prefix(format!("{}{}", prefix, ix));
                Q::build(vb, q_config.clone())
            })
            .collect();

        (varmap, qs)
    }

    pub fn soft_update(&mut self) -> Result<()> {
        track_with_replace_substring(
            &self.varmap_tgt,
            &self.varmap,
            self.tau,
            ("critic", "critic_tgt"),
        )?;
        Ok(())
    }

    /// Returns action values of all critics.
    pub fn qvals(&self, obs: &Q::Input1, act: &Q::Input2) -> Vec<Tensor> {
        self.qs
            .iter()
            .map(|critic| {
                let q = critic.forward(obs, act).squeeze(D::Minus1).unwrap();
                // debug_assert_eq!(q.dims(), &[self.batch_size]);
                q
            })
            .collect()
    }

    /// Returns minimum action values of all target critics.
    pub fn qvals_min(&self, obs: &Q::Input1, act: &Q::Input2) -> Result<Tensor> {
        let qvals = self.qvals(obs, act);
        let qvals = Tensor::stack(&qvals, 0)?; // [batch_size, self.n_nets]
        let qvals_min = qvals.min(0)?.squeeze(D::Minus1)?; // [batch_size]
        Ok(qvals_min)
    }

    /// Returns minimum action values of all target critics.
    pub fn qvals_min_tgt(&self, obs: &Q::Input1, act: &Q::Input2) -> Result<Tensor> {
        let qvals: Vec<Tensor> = self
            .qs_tgt
            .iter()
            .map(|critic| {
                let q = critic.forward(obs, act).squeeze(D::Minus1).unwrap();
                // debug_assert_eq!(q.dims(), &[self.batch_size]);
                q
            })
            .collect();
        let qvals = Tensor::stack(&qvals, 0)?; // [batch_size, self.n_nets]
        let qvals_min = qvals.min(0)?.squeeze(D::Minus1)?; // [batch_size]
        Ok(qvals_min)
    }
}

impl<Q> Clone for MultiCritic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + Clone,
{
    fn clone(&self) -> Self {
        let tau = self.tau;
        let n_nets = self.n_nets;
        let device = self.device.clone();
        let q_config = self.q_config.clone();
        let opt_config = self.opt_config.clone();

        // Critic networks
        let (mut varmap, qs) = Self::build_critic_networks(&q_config, &device, n_nets, "critic");

        // Target networks
        let (mut varmap_tgt, qs_tgt) =
            Self::build_critic_networks(&q_config, &device, n_nets, "critic_tgt");

        // Optimizer, shared with critic networks
        let opt = opt_config.build(varmap.all_vars()).unwrap();

        // Copy variables
        varmap.clone_from(&self.varmap);
        varmap_tgt.clone_from(&self.varmap_tgt);

        Self {
            tau,
            n_nets,
            device,
            varmap,
            varmap_tgt,
            q_config,
            qs,
            qs_tgt,
            opt_config,
            opt,
        }
    }
}

impl<Q> MultiCritic<Q>
where
    Q: SubModel2<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize,
{
    /// Backward step for all variables in critic networks.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.opt.backward_step(loss)
    }

    /// Save variables to prefix + ".pt" and + "_tgt.pt".
    pub fn save<T: AsRef<Path>>(&self, prefix: T) -> Result<(PathBuf, PathBuf)> {
        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("pt");
        self.varmap.save(&path.as_path())?;
        info!("Save critics to {:?}", path);

        let mut path_tgt = PathBuf::from(prefix.as_ref());
        path_tgt.set_extension("tgt.pt");
        self.varmap.save(&path_tgt.as_path())?;
        info!("Save target critics to {:?}", path_tgt);

        Ok((path, path_tgt))
    }

    /// Load variables from prefix + ".pt" and + "_tgt.pt".
    pub fn load<T: AsRef<Path>>(&mut self, prefix: T) -> Result<()> {
        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("pt");
        self.varmap.load(&path.as_path())?;
        info!("Load critics from {:?}", path);

        let mut path = PathBuf::from(prefix.as_ref());
        path.set_extension("tgt.pt");
        self.varmap.load(&path.as_path())?;
        info!("Load target critics from {:?}", path);

        Ok(())
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    /// Check variable names in a VarMap.
    fn test_varmap() -> Result<()> {
        let varmap = VarMap::new();

        // network 1
        let vb1 = VarBuilder::from_varmap(&varmap, F32, &Device::Cpu).set_prefix("critic1");
        candle_nn::linear(4, 4, vb1.pp("layer1"))?;
        candle_nn::linear(4, 4, vb1.pp("layer2"))?;

        // network 2
        let vb2 = VarBuilder::from_varmap(&varmap, F32, &Device::Cpu).set_prefix("critic2");
        candle_nn::linear(4, 4, vb2.pp("layer1"))?;
        candle_nn::linear(4, 4, vb2.pp("layer2"))?;

        // show variables in VarMap
        varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .for_each(|(key, _)| println!("{:?}", key));

        // Output:
        // --- iql::critic::test::test_varmap stdout ----
        // "critic1.layer1.weight"
        // "critic1.layer1.bias"
        // "critic2.layer1.bias"
        // "critic2.layer2.weight"
        // "critic2.layer2.bias"
        // "critic1.layer2.weight"
        // "critic1.layer2.bias"
        // "critic2.layer1.weight"

        Ok(())
    }

    #[test]
    /// Check broadcast on Tensor::lt().
    fn test_lt() -> Result<()> {
        use std::convert::TryFrom;

        // A scalar
        let tau = Tensor::try_from(0.7f32)?;

        // A vector
        let u = Tensor::from_slice(&[0.2f32, -0.1, 0.0, 0.02, -10.0], &[5], &Device::Cpu)?;

        // Expectile loss weight
        // let w = &tau - &u.lt(0f32)?;
        let w = &tau
            .broadcast_sub(&u.lt(0f32)?.to_dtype(candle_core::DType::F32)?)?
            .abs()?;

        println!("{:?}", tau.dims());
        println!("{:?}", w);

        Ok(())
    }
}
