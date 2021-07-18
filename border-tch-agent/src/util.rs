//! Utilities.
use crate::model::ModelBase;
use log::trace;
use serde::{Deserialize, Serialize};
mod mlp;
mod quantile_loss;
pub use mlp::{create_actor, create_critic, MLPConfig, MLP, MLP2};
pub use quantile_loss::quantile_huber_loss;

/// Interval between optimization steps.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum OptInterval {
    /// Optimization interval specified as interaction steps.
    Steps(usize),
    /// Optimization interval specified as episodes.
    Episodes(usize),
}

impl OptInterval {
    /// Constructs the counter for optimization.
    pub fn counter(self) -> OptIntervalCounter {
        OptIntervalCounter {
            opt_interval: self,
            count: 0,
        }
    }
}

/// The counter for optimization.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct OptIntervalCounter {
    opt_interval: OptInterval,
    count: usize,
}

impl OptIntervalCounter {
    /// Returns true if the optimization should be done.
    pub fn do_optimize(&mut self, is_done: &[i8]) -> bool {
        let is_done_any = is_done.iter().fold(0, |x, v| x + *v as i32) > 0;
        match self.opt_interval {
            OptInterval::Steps(interval) => {
                self.count += 1;
                if self.count == interval {
                    self.count = 0;
                    true
                } else {
                    false
                }
            }
            OptInterval::Episodes(interval) => {
                if is_done_any {
                    self.count += 1;
                    if self.count == interval {
                        self.count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}

/// Critic loss type.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum CriticLoss {
    /// Mean squared error.
    MSE,

    /// Smooth L1 loss.
    SmoothL1,
}

/// Apply soft update on a model.
///
/// Variables are identified by their names.
pub fn track<M: ModelBase>(dest: &mut M, src: &mut M, tau: f64) {
    let src = &mut src.get_var_store().variables();
    let dest = &mut dest.get_var_store().variables();
    debug_assert_eq!(src.len(), dest.len());

    let names = src.keys();
    tch::no_grad(|| {
        for name in names {
            let src = src.get(name).unwrap();
            let dest = dest.get_mut(name).unwrap();
            dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
        }
    });
    trace!("soft update");
}

/// Concatenates slices.
pub fn concat_slices(s1: &[i64], s2: &[i64]) -> Vec<i64> {
    let mut v = Vec::from(s1);
    v.append(&mut Vec::from(s2));
    v
}

/// Returns the dimension of output vectors, i.e., the number of discrete outputs.
pub trait OutDim {
    /// Returns the dimension of output vectors, i.e., the number of discrete outputs.
    fn get_out_dim(&self) -> i64;

    /// Sets the  output dimension.
    fn set_out_dim(&mut self, v: i64);
}
