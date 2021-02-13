use std::fmt::Debug;
use std::default::Default;
use ndarray::Array1;
use pyo3::{PyObject, IntoPy};

use crate::{
    core::{Act, record::{Record, RecordValue}},
    env::py_gym_env::PyGymEnvActFilter
};

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteAct {
    pub(crate) act: Vec<i32>,
}

impl PyGymEnvDiscreteAct {
    pub fn new(act: Vec<i32>) -> Self {
        Self {
            act,
        }
    }
}

impl Act for PyGymEnvDiscreteAct {}

// /// Filter action before applied to the environment.
// ///
// /// [`Record`] in the return value has `act`, which is a discrete action.
// pub trait PyGymDiscreteActFilter: Clone + Debug {
//     fn filt(act: Vec<i32>) -> (Vec<i32>, Record) {
//         let act_f32: Array1<f32> = act.iter().map(|v| *v as f32).collect();
//         (act, Record::from_slice(&[
//             ("act", RecordValue::Array1(act_f32.into()))
//         ]))
//     }
// }

#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteActRawFilter {
    pub vectorized: bool
}

impl Default for PyGymEnvDiscreteActRawFilter {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

impl PyGymEnvDiscreteActRawFilter {}

// TODO: support vecenv
impl PyGymEnvActFilter<PyGymEnvDiscreteAct> for PyGymEnvDiscreteActRawFilter {
    fn filt(&mut self, act: PyGymEnvDiscreteAct) -> (PyObject, Record) {
        let record = Record::from_slice(&[
            ("act", RecordValue::Array1(
                act.act.iter().map(|v| *v as f32).collect::<Vec<_>>().into()
            ))
        ]);

        let act = if self.vectorized {
            // TODO: Consider how to make Record object for vectorized environemnt
            unimplemented!();
            // pyo3::Python::with_gil(|py| {
            //     act.act.into_py(py)
            // })
        }
        else {
            pyo3::Python::with_gil(|py| {
                act.act[0].into_py(py)
            })
        };
        (act, record)
    }
}
