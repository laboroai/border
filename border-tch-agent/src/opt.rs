//! Optimizers.
use anyhow::Result;
use core::f64;
use serde::{Deserialize, Serialize};
use tch::{
    // nn,
    nn::{Adam, AdamW, Optimizer as Optimizer_, OptimizerConfig as OptimizerConfig_, VarStore},
    Tensor,
};

/// Configures an optimizer for training neural networks in an RL agent.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum OptimizerConfig {
    /// Adam optimizer.
    Adam {
        /// Learning rate.
        lr: f64,
    },

    AdamW {
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    },
}

impl OptimizerConfig {
    /// Constructs an optimizer.
    pub fn build(&self, vs: &VarStore) -> Result<Optimizer> {
        match &self {
            OptimizerConfig::Adam { lr } => {
                let opt = Adam::default().build(vs, *lr)?;
                Ok(Optimizer::Adam(opt))
            }
            OptimizerConfig::AdamW {
                lr,
                beta1,
                beta2,
                wd,
                eps,
                amsgrad,
            } => {
                let opt = AdamW {
                    beta1: *beta1,
                    beta2: *beta2,
                    wd: *wd,
                    eps: *eps,
                    amsgrad: *amsgrad,
                }
                .build(vs, *lr)?;
                Ok(Optimizer::AdamW(opt))
            }
        }
    }
}

/// Optimizers.
///
/// This is a thin wrapper of [tch::nn::Optimizer].
/// 
/// [tch::nn::Optimizer]: https://docs.rs/tch/0.16.0/tch/nn/struct.Optimizer.html
pub enum Optimizer {
    /// Adam optimizer.
    Adam(Optimizer_),

    AdamW(Optimizer_),
}

impl Optimizer {
    /// Applies a backward step pass.
    pub fn backward_step(&mut self, loss: &Tensor) {
        match self {
            Self::Adam(opt) => {
                opt.backward_step(loss);
            }
            Self::AdamW(opt) => {
                opt.backward_step(loss);
            }
        }
    }
}
