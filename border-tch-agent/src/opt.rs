//! Optimizers.
use anyhow::Result;
use core::f64;
use serde::{Deserialize, Serialize};
use tch::{
    // nn,
    nn::{Adam, Optimizer as Optimizer_, OptimizerConfig as OptimizerConfig_, VarStore},
    Tensor,
};

/// Configures an optimizer for training neural networks in an RL agent.
#[cfg(not(feature = "adam_eps"))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum OptimizerConfig {
    /// Adam optimizer.
    Adam {
        /// Learning rate.
        lr: f64,
    },
}

/// Configures an optimizer for training neural networks in an RL agent.
#[cfg(feature = "adam_eps")]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum OptimizerConfig {
    /// Adam optimizer.
    Adam {
        /// Learning rate.
        lr: f64,
    },

    /// Adam optimizer with the epsilon parameter.
    AdamEps {
        /// Learning rate.
        lr: f64,
        /// Epsilon parameter.
        eps: f64,
    },
}

#[cfg(not(feature = "adam_eps"))]
impl OptimizerConfig {
    /// Constructs an optimizer.
    pub fn build(&self, vs: &VarStore) -> Result<Optimizer> {
        match &self {
            OptimizerConfig::Adam { lr } => {
                let opt = Adam::default().build(vs, *lr)?;
                Ok(Optimizer::Adam(opt))
            }
        }
    }
}

#[cfg(feature = "adam_eps")]
impl OptimizerConfig {
    /// Constructs an optimizer.
    pub fn build(&self, vs: &VarStore) -> Result<Optimizer> {
        match &self {
            OptimizerConfig::Adam { lr } => {
                let opt = Adam::default().build(vs, *lr)?;
                Ok(Optimizer::Adam(opt))
            }
            OptimizerConfig::AdamEps { lr, eps } => {
                let mut opt = Adam::default();
                opt.eps = *eps;
                let opt = opt.build(vs, *lr)?;
                Ok(Optimizer::Adam(opt))
            }
        }
    }
}

/// Optimizers.
///
/// This is a thin wrapper of [tch::nn::Optimizer].
pub enum Optimizer {
    /// Adam optimizer.
    Adam(Optimizer_),
}

impl Optimizer {
    /// Applies a backward step pass.
    pub fn backward_step(&mut self, loss: &Tensor) {
        match self {
            Self::Adam(opt) => {
                opt.backward_step(loss);
            }
        }
    }
}
