//! Utinity functions for Python and Rust interoperation.
#[cfg(feature = "candle")]
pub mod candle;

pub mod ndarray {
    use ndarray::ArrayD;
    use num_traits::cast::AsPrimitive;
    use numpy::{Element, PyArrayDyn};
    use pyo3::{IntoPy, PyObject};

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

            // // Insert sample dimension
            // let obs = obs.insert_axis(Axis(0));

            obs
        })
    }

    /// Converts [`ArrayD<f32>`] to [`PyObject`].
    ///
    /// The type of the output array is `f64`.
    /// The first dimension of the input array, as expected to be a batch dimension, is removed.
    /// This function does not support batch action.
    pub fn arrayd_to_pyobj(act: ArrayD<f32>) -> PyObject {
        // let act = act.remove_axis(ndarray::Axis(0));
        pyo3::Python::with_gil(|py| {
            let act = act.mapv(f64::from);
            let act = PyArrayDyn::<f64>::from_array(py, &act);
            act.into_py(py)
        })
    }
}

pub mod vec {
    use anyhow::Result;
    use pyo3::{types::PyIterator, FromPyObject, PyAny, Python};

    pub fn pyany_to_f32vec(py: Python, pyany: &PyAny) -> Result<Vec<f32>> {
        let iter = PyIterator::from_object(py, pyany)?.iter()?;
        let vec = iter.map(|x| Ok(x?.extract::<f32>()?)).collect();
        vec
    }

    pub fn pyany_to_vec<'a, D: FromPyObject<'a>>(py: Python<'a>, pyany: &PyAny) -> Result<Vec<D>> {
        let iter = PyIterator::from_object(py, pyany)?.iter()?;
        let vec = iter.map(|x| Ok(x?.extract::<D>()?)).collect();
        vec
    }
}
