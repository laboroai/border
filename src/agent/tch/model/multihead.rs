use std::{path::Path, error::Error};
use log::{info, trace};
use tch::{Tensor, nn, nn::Module, Device, nn::OptimizerConfig};

use crate::agent::tch::model::{ModelBase, Model1};

#[derive(Debug)]
pub struct StateValueAndDiscreteActProb {
    var_store: nn::VarStore,
    network: nn::Sequential,
    actor: nn::Linear,
    critic: nn::Linear,
    device: Device,
    opt: nn::Optimizer<nn::Adam>,
}

impl StateValueAndDiscreteActProb {
    pub fn new(in_dim: usize, n_act: usize, learning_rate: f64) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let p = &vs.root();
        let network = nn::seq()
            .add(nn::linear(p / "cl1", in_dim as _, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "cl3", 256, 256 as _, Default::default()));
        let critic = nn::linear(p / "cl", 256, 1, Default::default());
        let actor = nn::linear(p / "al", 256, n_act as _, Default::default());
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();
        Self {
            network,
            actor,
            critic,
            device: p.device(),
            var_store: vs,
            opt,
        }
    }
}

impl ModelBase for StateValueAndDiscreteActProb {
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save StateValueAndDiscreteActProb model to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        };
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load StateValueAndDiscreteActProb model from {:?}", path.as_ref());
        Ok(())
    }
   
}

impl Model1 for StateValueAndDiscreteActProb {
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let shared = self.network.forward(xs);
        (shared.apply(&self.critic), shared.apply(&self.actor))
    }
}