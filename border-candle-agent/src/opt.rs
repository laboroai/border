//! Optimizers.
use anyhow::Result;
use candle_core::{Tensor, Var};
use candle_nn::{AdamW, Optimizer as _, ParamsAdamW};
use candle_optimisers::adam::{Adam, ParamsAdam};
use serde::{Deserialize, Serialize};

/// Configuration of optimizer for training neural networks in an RL agent.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum OptimizerConfig {
    /// AdamW optimizer.
    AdamW {
        lr: f64,
        #[serde(default = "default_beta1")]
        beta1: f64,
        #[serde(default = "default_beta2")]
        beta2: f64,
        #[serde(default = "default_eps")]
        eps: f64,
        #[serde(default = "default_weight_decay")]
        weight_decay: f64,
    },

    /// Adam optimizer.
    Adam {
        /// Learning rate.
        lr: f64,
    }
}

fn default_beta1() -> f64 {
    ParamsAdamW::default().beta1
}

fn default_beta2() -> f64 {
    ParamsAdamW::default().beta2
}

fn default_eps() -> f64 {
    ParamsAdamW::default().eps
}

fn default_weight_decay() -> f64 {
    ParamsAdamW::default().weight_decay
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
            OptimizerConfig::Adam { lr } => {
                let params = ParamsAdam {
                    lr: *lr,
                    ..ParamsAdam::default()
                };
                let opt = Adam::new(vars, params)?;
                Ok(Optimizer::Adam(opt))
            }
        }
    }

    /// Override learning rate.
    pub fn learning_rate(self, lr: f64) -> Self {
        match self {
            Self::AdamW {
                lr: _,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => Self::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            },
            Self::Adam {
                lr: _
            } => Self::Adam {
                lr
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

    Adam(Adam)
}

impl Optimizer {
    /// Applies a backward step pass.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        match self {
            Self::AdamW(opt) => Ok(opt.backward_step(loss)?),
            Self::Adam(opt) => Ok(opt.backward_step(loss)?),
        }
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        match self {
            Self::AdamW(opt) => Ok(opt.step(grads)?),
            Self::Adam(opt) => Ok(opt.step(grads)?),
        }
    }
}
