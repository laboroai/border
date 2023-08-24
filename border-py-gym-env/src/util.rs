use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};

/// Convert [`ArrayD<f32>`] to [`PyObject`].
///
/// This function does not support batch action.
pub fn arrayd_to_pyobj(act: ArrayD<f32>) -> PyObject {
    // let act = act.remove_axis(ndarray::Axis(0));
    pyo3::Python::with_gil(|py| {
        let act = PyArrayDyn::<f32>::from_array(py, &act);
        act.into_py(py)
    })
}

#[cfg(feature = "tch")]
use {std::convert::TryFrom, tch::Tensor};

#[cfg(feature = "tch")]
pub fn vec_to_tensor<T1, T2>(v: Vec<T1>, add_batch_dim: bool) -> Tensor
where
    T1: num_traits::AsPrimitive<T2>,
    T2: Copy + 'static + tch::kind::Element,
{
    let v = v.iter().map(|e| e.as_()).collect::<Vec<_>>();
    let t: Tensor = TryFrom::<Vec<T2>>::try_from(v).unwrap();

    match add_batch_dim {
        true => t.unsqueeze(0),
        false => t,
    }
}

#[cfg(feature = "tch")]
pub fn arrayd_to_tensor<T1, T2>(a: ArrayD<T1>, add_batch_dim: bool) -> Tensor
where
    T1: num_traits::AsPrimitive<T2>,
    T2: Copy + 'static + tch::kind::Element,
{
    let v = a.iter().map(|e| e.as_()).collect::<Vec<_>>();
    let t: Tensor = TryFrom::<Vec<T2>>::try_from(v).unwrap();

    match add_batch_dim {
        true => t.unsqueeze(0),
        false => t,
    }
}

#[cfg(feature = "tch")]
pub fn tensor_to_arrayd<T>(t: Tensor, delete_batch_dim: bool) -> ArrayD<T>
where
    T: tch::kind::Element
{
    let shape = match delete_batch_dim {
        false => t.size()[..].iter().map(|x| *x as usize).collect::<Vec<_>>(),
        true => t.size()[1..]
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>(),
    };
    let v: Vec<T> = t.into();

    ndarray::Array1::<T>::from(v)
        .into_shape(ndarray::IxDyn(&shape))
        .unwrap()
}
