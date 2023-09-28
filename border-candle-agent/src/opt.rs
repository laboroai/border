//! Optimizers.
use anyhow::Result;
use candle_core::{Tensor, Var};
use candle_nn::{AdamW, Optimizer as _, ParamsAdamW};
use serde::{Deserialize, Serialize};

/// Configuration of optimizer for training neural networks in an RL agent.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum OptimizerConfig {
    /// AdamW optimizer.
    AdamW {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
}

impl OptimizerConfig {
    /// Constructs [`AdamW`] optimizer.
    pub fn build(&self, vars: Vec<Var>) -> Result<Optimizer> {
        match &self {
            OptimizerConfig::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                let params = ParamsAdamW {
                    lr: *lr,
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                };
                let opt = AdamW::new(vars, params)?;
                Ok(Optimizer::AdamW(opt))
            }
        }
    }

    /// Override learning rate.
    pub fn learning_rate(self, lr: f64) -> Self {
        match self {
            Self::AdamW { lr: _, beta1, beta2, eps, weight_decay } => {
                Self::AdamW {
                    lr, beta1, beta2, eps, weight_decay
                }
            }
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        let params = ParamsAdamW::default();
        Self::AdamW {
            lr: params.lr,
            beta1: params.beta1,
            beta2: params.beta2,
            eps: params.eps,
            weight_decay: params.weight_decay,
        }
    }
}

/// Optimizers.
///
/// This is a thin wrapper of [tch::nn::Optimizer].
pub enum Optimizer {
    /// Adam optimizer.
    AdamW(AdamW),
}

impl Optimizer {
    /// Applies a backward step pass.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        match self {
            Self::AdamW(opt) => Ok(opt.backward_step(loss)?),
        }
    }
}
