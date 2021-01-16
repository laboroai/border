use std::{path::Path, error::Error};
use log::{info, trace};
use tch::{Tensor, nn, nn::Module, Device, nn::OptimizerConfig};
use crate::agents::tch::model::{ModelBase, Model2};

#[derive(Debug)]
pub struct Model2_1 {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    opt: nn::Optimizer<nn::Adam>,
    in_dim: usize,
    out_dim: usize,
    learning_rate: f64
}

impl Clone for Model2_1 {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.in_dim, self.out_dim, self.learning_rate);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Model2_1 {
    pub fn new(in_dim: usize, out_dim: usize, learning_rate: f64) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let p = &vs.root();
        let network = nn::seq()
            .add(nn::linear(p / "cl1", in_dim as _, 100, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "cl2", 100, out_dim as _, Default::default()));
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();
        Self {
            network,
            device: p.device(),
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
        let xs = Tensor::cat(&[x1s, x2s], -1);
        self.network.forward(&xs)
    }
}