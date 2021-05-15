//! DQN model.
use super::super::{
    model::{ModelBase, SubModel},
    opt::{Optimizer, OptimizerConfig},
};
use anyhow::Result;
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
/// Constructs [DQNModel].
pub struct DQNModelBuilder<F>
where
    F: SubModel,
    F::Config: DeserializeOwned + Serialize,
{
    feature_dim: i64,
    f_config: Option<F::Config>,
    opt_config: OptimizerConfig,
    phantom: PhantomData<F>,
}

impl<F: SubModel> Default for DQNModelBuilder<F>
where
    F::Config: DeserializeOwned + Serialize,
{
    fn default() -> Self {
        Self {
            feature_dim: 0,
            f_config: None,
            opt_config: OptimizerConfig::Adam { lr: 0.0 },
            phantom: PhantomData,
        }
    }
}

impl<F: SubModel> DQNModelBuilder<F>
where
    F::Config: DeserializeOwned + Serialize,
{
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

    /// Sets optimizer configuration.
    pub fn opt_config(mut self, v: OptimizerConfig) -> Self {
        self.opt_config = v;
        self
    }

    /// Constructs [DQNModelBuilder] from YAML file.
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

#[allow(clippy::upper_case_acronyms)]
/// Represents value functions for DQN agents.
pub struct DQNModel<Q>
where
    Q: SubModel<Output = Tensor>,
{
    device: Device,
    var_store: nn::VarStore,

    // Dimension of the input (feature) vector.
    // The `size()[-1]` of F::Output (Tensor) is feature_dim.
    feature_dim: i64,

    // Dimension of the output vector (equal to the number of actions).
    pub(super) out_dim: i64,

    // Action-value function
    q: Q,

    // Optimizer
    opt_config: OptimizerConfig,
    opt: Optimizer,

    phantom: PhantomData<Q>,
}

impl<Q> DQNModel<Q>
where
    Q: SubModel<Output = Tensor>,
{
    fn _build(
        &self,
        device: Device,
        feature_dim: i64,
        out_dim: i64,
        opt_config: OptimizerConfig,
        mut var_store: nn::VarStore,
    ) -> Self {
        // Action value function
        let q = self.q.clone_with_var_store(&var_store);

        // Optimizer
        let opt = opt_config.build(&var_store).unwrap();

        var_store.copy(&self.var_store).unwrap();

        Self {
            device,
            feature_dim,
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
}

impl<Q> Clone for DQNModel<Q>
where
    Q: SubModel<Output = Tensor>,
{
    fn clone(&self) -> Self {
        let device = self.device;
        let feature_dim = self.feature_dim;
        let out_dim = self.out_dim;
        let opt_config = self.opt_config.clone();
        let var_store = nn::VarStore::new(device);

        self._build(device, feature_dim, out_dim, opt_config, var_store)
    }
}

impl<Q> ModelBase for DQNModel<Q>
where
    Q: SubModel<Output = Tensor>,
{
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save DQN model to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load DQN model from {:?}", path.as_ref());
        Ok(())
    }
}
