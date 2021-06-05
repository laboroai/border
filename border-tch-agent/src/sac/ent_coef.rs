//! Entropy coefficient of SAC.
use log::{info, trace};
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, error::Error, path::Path};
use tch::{nn, nn::OptimizerConfig, Tensor};

/// Mode of the entropy coefficient of SAC.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum EntCoefMode {
    /// Use a constant as alpha.
    Fix(f64),
    /// Automatic tuning given `(target_entropy, learning_rate)`.
    Auto(f64, f64),
}

/// The entropy coefficient of SAC.
pub struct EntCoef {
    var_store: nn::VarStore,
    log_alpha: Tensor,
    target_entropy: Option<f64>,
    opt: Option<nn::Optimizer<nn::Adam>>,
}

impl EntCoef {
    /// Constructs an instance of `EntCoef`.
    pub fn new(mode: EntCoefMode, device: tch::Device) -> Self {
        let var_store = nn::VarStore::new(device);
        let path = &var_store.root();
        let (log_alpha, target_entropy, opt) = match mode {
            EntCoefMode::Fix(alpha) => {
                let init = nn::Init::Const(alpha.ln());
                let log_alpha = path.borrow().var("log_alpha", &[1], init);
                (log_alpha, None, None)
            }
            EntCoefMode::Auto(target_entropy, learning_rate) => {
                let init = nn::Init::Const(0.0);
                let log_alpha = path.borrow().var("log_alpha", &[1], init);
                let opt = nn::Adam::default()
                    .build(&var_store, learning_rate)
                    .unwrap();
                (log_alpha, Some(target_entropy), Some(opt))
            }
        };

        Self {
            var_store,
            log_alpha,
            opt,
            target_entropy,
        }
    }

    /// Returns the entropy coefficient.
    pub fn alpha(&self) -> Tensor {
        self.log_alpha.detach().exp()
    }

    /// Does an optimization step given a loss.
    pub fn backward_step(&mut self, loss: &Tensor) {
        if let Some(opt) = &mut self.opt {
            opt.backward_step(loss);
        }
    }

    /// Update the parameter given an action probability vector.
    pub fn update(&mut self, logp: &Tensor) {
        if let Some(target_entropy) = &self.target_entropy {
            let target_entropy = Tensor::from(*target_entropy);
            let loss = -(&self.log_alpha * (logp + target_entropy).detach()).mean(tch::Kind::Float);
            self.backward_step(&loss);
        }
    }

    /// Save the parameter into a file.
    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save entropy coefficient to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        }
        Ok(())
    }

    /// Save the parameter from a file.
    pub fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load entropy coefficient from {:?}", path.as_ref());
        Ok(())
    }
}
