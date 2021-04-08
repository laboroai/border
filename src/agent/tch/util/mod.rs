//! Utilities used by tch agents.
use log::trace;
use ndarray::ArrayD;
use tch::{Tensor, TchError};

use crate::agent::tch::model::ModelBase;

pub mod quantile_loss;
pub use quantile_loss::quantile_huber_loss;

/// Apply soft update on a model. 
pub fn track<M: ModelBase>(dest: &mut M, src: &mut M, tau: f64) {
    let src = &mut src.get_var_store();
    let dest = &mut dest.get_var_store();
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
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
