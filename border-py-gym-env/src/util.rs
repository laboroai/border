use ndarray::{concatenate, ArrayD, Axis};
use num_traits::cast::AsPrimitive;
use numpy::{Element, PyArrayDyn};
use pyo3::{IntoPy, PyObject};
use serde::{Deserialize, Serialize};

/// Converts PyObject to ArrayD.
pub fn pyobj_to_arrayd<T1, T2>(obs: PyObject) -> ArrayD<T2>
where
    T1: Element + AsPrimitive<T2>,
    T2: 'static + Copy,
{
    pyo3::Python::with_gil(|py| {
        let obs: &PyArrayDyn<T1> = obs.extract(py).unwrap();
        let obs = obs.to_owned_array();
        let obs = obs.mapv(|elem| elem.as_());

        // Insert sample dimension
        let obs = obs.insert_axis(Axis(0));

        obs
    })
}

/// Converts [`ArrayD<f32>`] to [`PyObject`].
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
mod _tch {
    use super::*;
    use {std::convert::TryFrom, tch::Tensor};

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

    pub fn tensor_to_arrayd<T>(t: Tensor, delete_batch_dim: bool) -> ArrayD<T>
    where
        T: tch::kind::Element,
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
}

#[cfg(feature = "tch")]
pub use _tch::*;

#[cfg(feature = "candle-core")]
mod _candle {
    use super::*;
    use anyhow::Result;
    use candle_core::{Tensor, WithDType};
    use std::convert::TryFrom;

    pub fn vec_to_tensor<T1, T2>(v: Vec<T1>, add_batch_dim: bool) -> Result<Tensor>
    where
        T1: num_traits::AsPrimitive<T2>,
        T2: WithDType,
    {
        let v = v.iter().map(|e| e.as_()).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<T2>>::try_from(v).unwrap();

        match add_batch_dim {
            true => Ok(t.unsqueeze(0)?),
            false => Ok(t),
        }
    }

    pub fn arrayd_to_tensor<T1, T2>(a: ArrayD<T1>, add_batch_dim: bool) -> Result<Tensor>
    where
        T1: num_traits::AsPrimitive<T2>,
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

    pub fn tensor_to_arrayd<T>(t: Tensor, delete_batch_dim: bool) -> Result<ArrayD<T>>
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
}

#[cfg(feature = "candle-core")]
pub use _candle::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ArrayType {
    F32Array,
}

impl ArrayType {
    pub fn to_array(&self, pyobj: PyObject) -> Array {
        match &self {
            Self::F32Array => Array::F32Array(pyobj_to_arrayd::<f32, f32>(pyobj)),
        }
    }
}

#[derive(Clone, Debug)]
/// Wraps [`ArrayD`] with concrete data type.
pub enum Array {
    Empty,
    F32Array(ArrayD<f32>),
}

impl Array {
    pub fn to_flat_vec<T>(&self) -> Vec<T>
    where
        f32: AsPrimitive<T>,
        T: Copy + 'static,
    {
        match self {
            Self::F32Array(array) => array.iter().map(|x| x.as_()).collect(),
            Self::Empty => {
                panic!()
            } // TODO: better message
        }
    }

    /// Returns the size of data in the object.
    pub fn len(&self) -> usize {
        match self {
            Self::F32Array(array) => array.shape()[0],
            Self::Empty => {
                panic!()
            } // TODO: better message
        }
    }

    pub fn as_f32_array(self) -> ArrayD<f32> {
        match self {
            Self::F32Array(array) => array,
            _ => panic!(),
        }
    }

    /// Horizontally stacks elements in `arrays`.
    ///
    /// Data type are automatically changed to that of the first array.
    pub fn hstack(arrays: Vec<Self>) -> Self {
        let (datatype, last_axis) = match &arrays[0] {
            Self::Empty => panic!(),
            Self::F32Array(array) => (0, (array.shape().len() - 1)),
        };

        match datatype {
            0 => {
                let arrays_ = arrays
                    .into_iter()
                    .map(|array| array.as_f32_array())
                    .collect::<Vec<_>>();
                let arrays = arrays_.iter().map(|array| array.view()).collect::<Vec<_>>();

                // debug print
                // arrays.iter().for_each(|a| println!("{:?}", a.shape()));
                // println!("{:?}", last_axis);

                let array = concatenate(Axis(last_axis), &arrays[..]).unwrap();
                Self::F32Array(array)
            }
            _ => panic!(),
        }
    }
}
