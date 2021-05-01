//! Utility functions using tch-rs.
use ndarray::ArrayD;
use tch::{TchError, Tensor};

/// Converts [ndarray::ArrayD] into tch Tensor.
/// Borrowed from tch-rs. The original code didn't work with ndarray 0.14.
pub fn try_from<T>(value: ArrayD<T>) -> Result<Tensor, TchError>
where
    T: tch::kind::Element,
{
    // TODO: Replace this with `?` once it works with `std::option::ErrorNone`
    let slice = match value.as_slice() {
        None => return Err(TchError::Convert("cannot convert to slice".to_string())),
        Some(v) => v,
    };
    let tn = Tensor::f_of_slice(slice)?;
    let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
    // Ok(tn.f_reshape(&shape)?)
    tn.f_reshape(&shape)
}
