use std::{path::Path, error::Error, fmt, fmt::{Formatter, Debug}};
use log::{info, trace};
use tch::{Tensor, nn, nn::Module, nn::OptimizerConfig};

use crate::agent::tch::model::{ModelBase, Model2};

pub struct Model2_1 {
    var_store: nn::VarStore,
    network_fn: fn(&nn::Path, usize, usize) -> nn::Sequential,
    network: nn::Sequential,
    opt: nn::Optimizer<nn::Adam>,
    in_dim: usize,
    out_dim: usize,
    learning_rate: f64
}

/// TODO: implement debug print.
impl Debug for Model2_1 {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result { Ok(()) }
}

impl Clone for Model2_1 {
    fn clone(&self) -> Self {
        let device = self.var_store.device();
        let mut new = Self::new(self.in_dim, self.out_dim, self.learning_rate, self.network_fn, device);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Model2_1 {
    pub fn new(in_dim: usize, out_dim: usize, learning_rate: f64,
        network_fn: fn(&nn::Path, usize, usize) -> nn::Sequential, device: tch::Device) -> Self {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();
        let network = network_fn(p, in_dim, out_dim);
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();
        Self {
            network,
            network_fn,
            // device: p.device(),
            var_store: vs,
            in_dim,
            out_dim,
            opt,
            learning_rate,
        }
    }
}

impl ModelBase for Model2_1 {
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
        };
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load qnet from {:?}", path.as_ref());
        Ok(())
    }
}

impl Model2 for Model2_1 {
    type Input1 = Tensor;
    type Input2 = Tensor;
    type Output = Tensor;

    fn forward(&self, x1s: &Tensor, x2s: &Tensor) -> Tensor {
        let device = self.var_store.device();
        let x1s = x1s.to_device(device);
        let x2s = x2s.to_device(device);
        let xs = Tensor::cat(&[&x1s, &x2s], -1);
        self.network.forward(&xs)
    }
}