//! Continuous action for [PyGymEnv](crate::PyGymEnv).
mod base;
mod raw_filter;
pub use base::PyGymEnvContinuousAct;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};
pub use raw_filter::{PyGymEnvContinuousActRawFilter, PyGymEnvContinuousActRawFilterConfig};

/// Convert [`ArrayD<f32>`] to [`PyObject`].
///
/// This function does not support batch action.
pub fn to_pyobj(act: ArrayD<f32>) -> PyObject {
    // let act = act.remove_axis(ndarray::Axis(0));
    pyo3::Python::with_gil(|py| {
        let act = PyArrayDyn::<f32>::from_array(py, &act);
        act.into_py(py)
    })
}
