use log::trace;
use ndarray::ArrayD;
use tch::{Tensor, TchError};

use crate::agent::tch::model::ModelBase;

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

// Borrowed from tch-rs. The original code didn't work with ndarray 0.14.
pub fn try_from(value: ArrayD<f32>) -> Result<Tensor, TchError> {
    // TODO: Replace this with `?` once it works with `std::option::ErrorNone`
    let slice = match value.as_slice() {
        None => return Err(TchError::Convert("cannot convert to slice".to_string())),
        Some(v) => v,
    };
    let tn = Tensor::f_of_slice(slice)?;
    let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
    Ok(tn.f_reshape(&shape)?)
}

pub fn sum_keep1(t: &Tensor) -> Tensor {
    let t_ = t.view([[t.size().as_slice()[0]], [-1i64]].concat().as_slice());
    t_.sum1(&[-1], false, tch::Kind::Float)
}

pub fn concat_slices(s1: &[i64], s2: &[i64]) -> Vec<i64> {
    let mut v = Vec::from(s1);
    v.append(&mut Vec::from(s2));
    v
}
