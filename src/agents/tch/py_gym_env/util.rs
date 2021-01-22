
use num_traits::cast::AsPrimitive;
use pyo3::{PyObject};
use ndarray::{ArrayD, Axis};
use numpy::{PyArrayDyn, Element};
use crate::agents::tch::Shape;

/// Convert PyObject to ArrayD.
///
/// If the shape of the PyObject has the number of axes equal to the shape of
/// observation, i.e., S.shape(), an axis is appended, corresponding to
/// the process number, as it comes from a vectorized environment with single process.
pub fn pyobj_to_arrayd<S, T>(obs: PyObject) -> ArrayD<f32> where
    S: Shape,
    T: Element + AsPrimitive<f32>,
{
    pyo3::Python::with_gil(|py| {
        let obs: &PyArrayDyn<T> = obs.extract(py).unwrap();
        let obs = obs.to_owned_array();
        // let obs = obs.mapv(|elem| elem as f32);
        let obs = obs.mapv(|elem| elem.as_());
        let obs = {
            if obs.shape().len() == S::shape().len() + 1 {
                // In this case obs has a dimension for n_procs
                obs
            }
            else if obs.shape().len() == S::shape().len() {
                // add dimension for n_procs
                obs.insert_axis(Axis(0))
            }
            else {
                panic!();
            }
        };
        obs
    })
}
