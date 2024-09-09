//! Utilities.
use crate::model::ModelBase;
use log::trace;
use serde::{Deserialize, Serialize};
mod named_tensors;
mod quantile_loss;
use border_core::record::{Record, RecordValue};
pub use named_tensors::NamedTensors;
use ndarray::ArrayD;
use num_traits::cast::AsPrimitive;
pub use quantile_loss::quantile_huber_loss;
use std::convert::TryFrom;
use tch::{nn::VarStore, Tensor};

/// Critic loss type.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum CriticLoss {
    /// Mean squared error.
    Mse,

    /// Smooth L1 loss.
    SmoothL1,
}

/// Apply soft update on variables.
///
/// Variables are identified by their names.
///
/// dest = tau * src + (1.0 - tau) * dest
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

/// Interface for handling output dimensions.
pub trait OutDim {
    /// Returns the output dimension.
    fn get_out_dim(&self) -> i64;

    /// Sets the  output dimension.
    fn set_out_dim(&mut self, v: i64);
}

/// Returns the mean and standard deviation of the parameters.
pub fn param_stats(var_store: &VarStore) -> Record {
    let mut record = Record::empty();

    for (k, v) in var_store.variables() {
        // let m: f32 = v.mean(tch::Kind::Float).into();
        let m = f32::try_from(v.mean(tch::Kind::Float)).expect("Failed to convert Tensor to f32");
        let k_mean = format!("{}_mean", &k);
        record.insert(k_mean, RecordValue::Scalar(m));

        let m = f32::try_from(v.std(false)).expect("Failed to convert Tensor to f32");
        let k_std = format!("{}_std", k);
        record.insert(k_std, RecordValue::Scalar(m));
    }

    record
}

pub fn vec_to_tensor<T1, T2>(v: Vec<T1>, add_batch_dim: bool) -> Tensor
where
    T1: AsPrimitive<T2>,
    T2: Copy + 'static + tch::kind::Element,
{
    let v = v.iter().map(|e| e.as_()).collect::<Vec<_>>();
    let t: Tensor = TryFrom::<Vec<T2>>::try_from(v).unwrap();

    match add_batch_dim {
        true => t.unsqueeze(0),
        false => t,
    }
}

/// Converts [`ndarray::ArrayD`] to [`Tensor`].
pub fn arrayd_to_tensor<T1, T2>(a: ArrayD<T1>, add_batch_dim: bool) -> Tensor
where
    T1: AsPrimitive<T2>,
    T2: Copy + 'static + tch::kind::Element,
{
    let v = a.iter().map(|e| e.as_()).collect::<Vec<_>>();
    let t: Tensor = TryFrom::<Vec<T2>>::try_from(v).unwrap();

    match add_batch_dim {
        true => t.unsqueeze(0),
        false => t,
    }
}

/// Converts [`Tensor`] to [`ndarray::ArrayD`].
pub fn tensor_to_arrayd<T>(t: Tensor, delete_batch_dim: bool) -> ArrayD<T>
where
    T: tch::kind::Element + Copy,
{
    let shape = match delete_batch_dim {
        false => t.size()[..].iter().map(|x| *x as usize).collect::<Vec<_>>(),
        true => t.size()[1..]
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>(),
    };
    let v = Vec::<T>::try_from(&t.flatten(0, -1)).expect("Failed to convert from Tensor to Vec");

    ndarray::Array1::<T>::from(v)
        .into_shape(ndarray::IxDyn(&shape))
        .unwrap()
}
