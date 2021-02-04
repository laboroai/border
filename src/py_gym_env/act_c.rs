use std::fmt::Debug;
use std::marker::PhantomData;
use pyo3::{PyObject, IntoPy};
use ndarray::{Axis, ArrayD};
use numpy::PyArrayDyn;
use crate::core::Act;
use crate::agents::tch::Shape;

/// Filtering action before applied to the environment.
pub trait PyGymEnvContinuousActFilter: Clone + Debug {
    fn filter(act: ArrayD<f32>) -> ArrayD<f32> {
        act
    }
}

#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousActRawFilter {}

impl PyGymEnvContinuousActFilter for PyGymEnvContinuousActRawFilter {}

/// Represents action.
/// Currently, it supports 1-dimensional vector only.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousAct<S: Shape, F: PyGymEnvContinuousActFilter> {
    pub(crate) act: ArrayD<f32>,
    pub(crate) phantom: PhantomData<(S, F)>
}

impl<S: Shape, F: PyGymEnvContinuousActFilter> PyGymEnvContinuousAct<S, F> {
    pub fn new(v: ArrayD<f32>) -> Self {
        Self {
            act: v,
            phantom: PhantomData
        }
    }
}

impl<S: Shape, F: PyGymEnvContinuousActFilter> Act for PyGymEnvContinuousAct<S, F> {}

/// TODO: check action representation in the vectorized environment.
impl<S: Shape, F: PyGymEnvContinuousActFilter> Into<PyObject> for PyGymEnvContinuousAct<S, F> {
    fn into(self) -> PyObject {
        let act = (&self.act).clone();
        let act = {
            if S::squeeze_first_dim() {
                debug_assert_eq!(self.act.shape()[0], 1);
                let act = act.remove_axis(ndarray::Axis(0));
                let act = F::filter(act);
                pyo3::Python::with_gil(|py| {
                    let act = PyArrayDyn::<f32>::from_array(py, &act);
                    act.into_py(py)
                })        
            }
            else {
                // Consider the first axis as processes in vectorized environments
                pyo3::Python::with_gil(|py| {
                    act.axis_iter(Axis(0))
                        .map(|act| F::filter(act.to_owned()))
                        .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
                        .collect::<Vec<_>>().into_py(py)
                })
            }
        };
        act
    }
}
