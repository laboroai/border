//! Continuous action for [`super::PyGymEnv`] and [`super::PyVecGymEnv`].
mod base;
mod raw_filter;
pub use base::PyGymEnvContinuousAct;
pub use raw_filter::{PyGymEnvContinuousActRawFilter, PyGymEnvContinuousActRawFilterConfig};
use border_core::Shape;
use ndarray::{ArrayD, Axis};
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};

/// Convert [ArrayD<f32>] to [PyObject].
///
/// The first element of the shape of `act` is batch dimension and
/// `act.size()[1..]` is equal to S::shape().
///
/// TODO: explain how to handle the first dimension for vectorized environment.
pub fn to_pyobj<S: Shape>(act: ArrayD<f32>) -> PyObject {
    if S::squeeze_first_dim() {
        debug_assert_eq!(act.shape()[0], 1);
        debug_assert_eq!(&act.shape()[1..], S::shape());
        let act = act.remove_axis(ndarray::Axis(0));
        pyo3::Python::with_gil(|py| {
            let act = PyArrayDyn::<f32>::from_array(py, &act);
            act.into_py(py)
        })
    } else {
        // Interpret the first axis as processes in vectorized environments
        pyo3::Python::with_gil(|py| {
            act.axis_iter(Axis(0))
                .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
                .collect::<Vec<_>>()
                .into_py(py)
        })
    }
}
