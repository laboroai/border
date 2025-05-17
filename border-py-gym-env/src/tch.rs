mod ndarray_act;
mod ndarray_converter;
mod ndarray_obs;
mod tensor_batch;
pub use ndarray_act::NdarrayAct;
pub use ndarray_converter::{NdarrayConverter, NdarrayConverterConfig};
pub use ndarray_obs::NdarrayObs;
pub use tensor_batch::{TensorBatch, ZeroTensor};

use ndarray::ArrayD;
use num_traits::cast::AsPrimitive;
use std::convert::TryFrom;
use tch::Tensor;

/// Converts [`ArrayD`] to [`Tensor`].
///
/// [`ArrayD`]: [`ndarray::ArrayD`]
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

/// Converts [`Tensor`] to [`ArrayD`].
///
/// [`ArrayD`]: [`ndarray::ArrayD`]
pub fn tensor_to_arrayd<T>(t: Tensor, delete_batch_dim: bool) -> ArrayD<T>
where
    T: tch::kind::Element + Copy + std::fmt::Debug,
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
