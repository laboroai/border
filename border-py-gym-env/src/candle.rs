use anyhow::Result;
use candle_core::{Tensor, WithDType};
use ndarray::ArrayD;
use num_traits::AsPrimitive;
pub mod tensor;
mod tensor_batch;
use std::convert::TryFrom;
pub use tensor_batch::{TensorBatch, ZeroTensor};

fn arrayd_to_tensor<T1, T2>(a: ArrayD<T1>, add_batch_dim: bool) -> Result<Tensor>
where
    T1: AsPrimitive<T2>,
    T2: WithDType,
{
    let shape = a.shape();
    let v = a.iter().map(|e| e.as_()).collect::<Vec<_>>();
    let t: Tensor = TryFrom::<Vec<T2>>::try_from(v)?;
    let t = t.reshape(shape)?;

    match add_batch_dim {
        true => Ok(t.unsqueeze(0)?),
        false => Ok(t),
    }
}

fn tensor_to_arrayd<T>(t: Tensor, delete_batch_dim: bool) -> Result<ArrayD<T>>
where
    T: WithDType, //tch::kind::Element,
{
    let shape = match delete_batch_dim {
        false => t.dims()[..].iter().map(|x| *x as usize).collect::<Vec<_>>(),
        true => t.dims()[1..]
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>(),
    };
    let v: Vec<T> = t.flatten_all()?.to_vec1()?;

    Ok(ndarray::Array1::<T>::from(v).into_shape(ndarray::IxDyn(&shape))?)
}
