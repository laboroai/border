use std::fmt::Debug;
use std::default::Default;
use std::marker::PhantomData;
use pyo3::{PyObject, IntoPy};
use ndarray::{Axis, ArrayD};
use numpy::PyArrayDyn;
use crate::core::Act;
use crate::agents::tch::Shape;

use super::PyGymEnvActFilter;

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousAct<S: Shape> {
    pub(crate) act: ArrayD<f32>,
    pub(crate) phantom: PhantomData<S>
}

impl<S: Shape> PyGymEnvContinuousAct<S> {
    pub fn new(v: ArrayD<f32>) -> Self {
        Self {
            act: v,
            phantom: PhantomData
        }
    }
}

impl<S: Shape> Act for PyGymEnvContinuousAct<S> {}

/// Filtering action before applied to the environment.
pub trait PyGymEnvContinuousActFilter: Clone + Debug {
    fn filter(act: ArrayD<f32>) -> ArrayD<f32> {
        act
    }
}

#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousActRawFilter {
    pub vectorized: bool
}

impl Default for PyGymEnvContinuousActRawFilter {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

/// TODO: check action representation in the vectorized environment.
impl<S: Shape> PyGymEnvActFilter<PyGymEnvContinuousAct<S>> for PyGymEnvContinuousActRawFilter {
    fn filt(&mut self, act: PyGymEnvContinuousAct<S>) -> PyObject {
        let act = act.act;
        let act = {
            if S::squeeze_first_dim() {
                debug_assert_eq!(act.shape()[0], 1);
                debug_assert_eq!(&act.shape()[1..], S::shape());
                let act = act.remove_axis(ndarray::Axis(0));
                pyo3::Python::with_gil(|py| {
                    let act = PyArrayDyn::<f32>::from_array(py, &act);
                    act.into_py(py)
                })        
            }
            else {
                // Consider the first axis as processes in vectorized environments
                pyo3::Python::with_gil(|py| {
                    act.axis_iter(Axis(0))
                        .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
                        .collect::<Vec<_>>().into_py(py)
                })
            }
        };
        act
    }
}
