//! Utilities.
use crate::model::ModelBase;
use log::trace;
use serde::{Deserialize, Serialize};
mod quantile_loss;
mod named_tensors;
pub use quantile_loss::quantile_huber_loss;
pub use named_tensors::NamedTensors;

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
