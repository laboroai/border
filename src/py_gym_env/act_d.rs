use std::marker::PhantomData;
use std::fmt::Debug;
use pyo3::{PyObject, IntoPy};
use crate::core::Act;

/// Filter action before applied to the environment.
pub trait PyGymDiscreteActFilter: Clone + Debug {
    fn filt(act: Vec<i32>) -> Vec<i32> {
        act
    }
}

#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteActRawFilter {}

impl PyGymDiscreteActFilter for PyGymEnvDiscreteActRawFilter {}

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteAct<F: PyGymDiscreteActFilter> {
    pub(crate) act: Vec<i32>,
    pub(crate) phantom: PhantomData<F>
}

impl<F: PyGymDiscreteActFilter> PyGymEnvDiscreteAct<F> {
    pub fn new(act: Vec<i32>) -> Self {
        Self {
            act,
            phantom: PhantomData
        }
    }
}

impl<F: PyGymDiscreteActFilter> Act for PyGymEnvDiscreteAct<F> {}

impl<F: PyGymDiscreteActFilter> Into<PyObject> for PyGymEnvDiscreteAct<F> {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            F::filt(self.act).into_py(py)
        })
    }
}
