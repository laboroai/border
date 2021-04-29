//! Neural network with a single input tensor and double output tensors.
use log::{info, trace};
use std::{
    error::Error,
    fmt,
    fmt::{Debug, Formatter},
    path::Path,
};
use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor};

use crate::agent::tch::model::{Model1, ModelBase};

/// Neural network with a single input tensor and double output tensors.
///
/// This network model is used as an actor in [crate::agent::tch::sac::SAC].
pub struct Model1_2 {
    var_store: nn::VarStore,
    network_fn: fn(&nn::Path, usize, usize) -> nn::Sequential,
    network: nn::Sequential,
    opt: nn::Optimizer<nn::Adam>,
    head_mean: nn::Linear,
    head_lstd: nn::Linear,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    learning_rate: f64,
}

// TODO: implement this
impl Debug for Model1_2 {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl Clone for Model1_2 {
    fn clone(&self) -> Self {
        let device = self.var_store.device();
        let mut new = Self::new(
            self.in_dim,
            self.hidden_dim,
            self.out_dim,
            self.learning_rate,
            self.network_fn,
            device,
        );
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Model1_2 {
    /// Constructs a network.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        learning_rate: f64,
        network_fn: fn(&nn::Path, usize, usize) -> nn::Sequential,
        device: tch::Device,
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();
        let network = network_fn(p, in_dim, hidden_dim);
        let head_mean = nn::linear(p / "ml", hidden_dim as _, out_dim as _, Default::default());
        let head_lstd = nn::linear(p / "sl", hidden_dim as _, out_dim as _, Default::default());

        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();
        Self {
            network,
            network_fn,
            var_store: vs,
            in_dim,
            hidden_dim,
            out_dim,
            head_mean,
            head_lstd,
            opt,
            learning_rate,
        }
    }
}

impl ModelBase for Model1_2 {
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save qnet to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load qnet from {:?}", path.as_ref());
        Ok(())
    }
}

impl Model1 for Model1_2 {
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, xs: &Tensor) -> Self::Output {
        let device = self.var_store.device();
        let xs = xs.to_device(device);
        let xs = self.network.forward(&xs);
        (xs.apply(&self.head_mean), xs.apply(&self.head_lstd).exp())
    }

    fn in_shape(&self) -> &[usize] {
        unimplemented!()
        // self.in_dim
    }

    fn out_dim(&self) -> usize {
        self.out_dim
    }
}
