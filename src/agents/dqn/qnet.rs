use tch::{Tensor, nn, nn::Module, Device, nn::OptimizerConfig};
use crate::core::{Policy, Agent};

#[derive(Debug)]
pub struct QNetwork {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    opt: nn::Optimizer<nn::Adam>,
    in_dim: usize,
    out_dim: usize,
    learning_rate: f64
}

impl Clone for QNetwork {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.in_dim, self.out_dim, self.learning_rate);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl QNetwork {
    pub fn new(in_dim: usize, out_dim: usize, learning_rate: f64) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let opt = nn::Adam::default().build(&var_store, 1e-3).unwrap();
        let p = &var_store.root();
        Self {
            network: nn::seq()
                .add(nn::linear(
                    p / "cl1",
                    in_dim as _,
                    400,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "cl2", 400, 300, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "cl3", 300, out_dim as _, Default::default())),
            device: p.device(),
            var_store,
            in_dim,
            out_dim,
            opt,
            learning_rate,
        }
    }
}

impl Module for QNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.network.forward(xs)
    }
}
