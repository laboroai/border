//! Entropy coefficient of SAC.
use std::convert::TryFrom;

use crate::opt::{Optimizer, OptimizerConfig};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{init::Init, VarBuilder, VarMap};
use log::info;
use serde::{Deserialize, Serialize};
use std::path::Path;

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
    varmap: VarMap,
    log_alpha: Tensor,
    target_entropy: Option<f64>,
    opt: Option<Optimizer>,
}

impl EntCoef {
    /// Constructs an instance of `EntCoef`.
    pub fn new(mode: EntCoefMode, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let (log_alpha, target_entropy, opt) = match mode {
            EntCoefMode::Fix(alpha) => {
                let init = Init::Const(alpha.ln());
                let log_alpha = vb.get_with_hints(1, "log_alpha", init)?;
                (log_alpha, None, None)
            }
            EntCoefMode::Auto(target_entropy, learning_rate) => {
                let init = Init::Const(0.0);
                let log_alpha = vb.get_with_hints(1, "log_alpha", init)?;
                let opt = OptimizerConfig::default()
                    .learning_rate(learning_rate)
                    .build(varmap.all_vars())?;
                (log_alpha, Some(target_entropy), Some(opt))
            }
        };

        Ok(Self {
            varmap,
            log_alpha,
            opt,
            target_entropy,
        })
    }

    /// Returns the entropy coefficient.
    pub fn alpha(&self) -> Result<Tensor> {
        Ok(self.log_alpha.detach().exp()?)
    }

    /// Does an optimization step given a loss.
    pub fn backward_step(&mut self, loss: &Tensor) {
        if let Some(opt) = &mut self.opt {
            opt.backward_step(loss).unwrap();
        }
    }

    /// Update the parameter given an action probability vector.
    pub fn update(&mut self, logp: &Tensor) -> Result<()> {
        if let Some(target_entropy) = &self.target_entropy {
            let target_entropy = Tensor::try_from(*target_entropy)?;
            let loss = {
                let tmp = ((&self.log_alpha * (logp + target_entropy)?.detach())? * -1f64)?;
                tmp.mean(0)?
            };
            self.backward_step(&loss);
        }
        Ok(())
    }

    /// Save the parameter into a file.
    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        self.varmap.save(&path)?;
        info!("Save entropy coefficient to {:?}", path.as_ref());
        Ok(())
    }

    /// Save the parameter from a file.
    pub fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.varmap.load(&path)?;
        info!("Load entropy coefficient from {:?}", path.as_ref());
        Ok(())
    }
}
