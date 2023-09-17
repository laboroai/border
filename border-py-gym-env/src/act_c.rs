//! Continuous action for [`GymEnv`](crate::GymEnv).
mod base;
pub use base::GymContinuousAct;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};

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
