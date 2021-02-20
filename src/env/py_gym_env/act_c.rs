use std::fmt::Debug;
use std::default::Default;
use std::marker::PhantomData;
use pyo3::{PyObject, IntoPy};
use ndarray::{Axis, ArrayD};
use numpy::PyArrayDyn;

use crate::{
    core::{Act, record::{Record, RecordValue}},
    env::py_gym_env::Shape,
};

use super::PyGymEnvActFilter;

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousAct<S: Shape> {
    pub act: ArrayD<f32>,
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

#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousActRawFilter {
    pub vectorized: bool
}

impl Default for PyGymEnvContinuousActRawFilter {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

/// Convert [crate::env::py_gym_env::act_c::PyGymEnvContinuousAct] to `PyObject`.
/// No processing will be applied to the action.
///
/// TODO: explain action representation for the vectorized environment.
impl<S: Shape> PyGymEnvActFilter<PyGymEnvContinuousAct<S>> for PyGymEnvContinuousActRawFilter {
    fn filt(&mut self, act: PyGymEnvContinuousAct<S>) -> (PyObject, Record) {
        let act = act.act;
        let record = Record::from_slice(&[
            ("act", RecordValue::Array1(act.iter().cloned().collect())
            )
        ]);

        // TODO: replace the following code with to_pyobj()
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
                // Interpret the first axis as processes in vectorized environments
                pyo3::Python::with_gil(|py| {
                    act.axis_iter(Axis(0))
                        .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
                        .collect::<Vec<_>>().into_py(py)
                })
            }
        };
        (act, record)
    }
}

/// Convert `PyGymEnvContinuousAct` to `PyObject`.
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
    }
    else {
        // Interpret the first axis as processes in vectorized environments
        pyo3::Python::with_gil(|py| {
            act.axis_iter(Axis(0))
                .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
                .collect::<Vec<_>>().into_py(py)
        })
    }
}
