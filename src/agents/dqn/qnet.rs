use tch::{Tensor, nn, nn::Module, Device, nn::OptimizerConfig};
use crate::agents::Model;

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
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let p = &vs.root();
        let network = nn::seq()
            .add(nn::linear(
                p / "cl1",
                in_dim as _,
                400,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "cl2", 400, 300, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "cl3", 300, out_dim as _, Default::default()));
        let opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
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

impl Model for QNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.network.forward(xs)
    }

    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }
}
