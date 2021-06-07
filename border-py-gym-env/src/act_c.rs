//! Continuous action for [`super::PyGymEnv`] and [`super::PyVecGymEnv`].
use crate::PyGymEnvActFilter;
use border_core::{
    Act, Shape, record::{Record, RecordValue},
};
use pyo3::{IntoPy, PyObject};
use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;
use ndarray::{ArrayD, Axis};
use numpy::PyArrayDyn;

// use crate::env::py_gym_env::Shape;
// use border_core::{
//     record::{Record, RecordValue},
//     Act,
// };

/// Represents an action.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousAct<S: Shape> {
    /// Stores an action.
    pub act: ArrayD<f32>,
    pub(crate) phantom: PhantomData<S>,
}

impl<S: Shape> PyGymEnvContinuousAct<S> {
    /// Constructs an action.
    pub fn new(act: ArrayD<f32>) -> Self {
        Self {
            act,
            phantom: PhantomData,
        }
    }
}

impl<S: Shape> Act for PyGymEnvContinuousAct<S> {}

/// Raw filter for continuous actions.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousActRawFilter {
    /// `true` indicates that this filter is used in a vectorized environment.
    pub vectorized: bool,
}

impl Default for PyGymEnvContinuousActRawFilter {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

impl<S: Shape> PyGymEnvActFilter<PyGymEnvContinuousAct<S>> for PyGymEnvContinuousActRawFilter {
    /// Convert [PyGymEnvContinuousAct] to [PyObject].
    /// No processing will be applied to the action.
    ///
    /// The first element of the shape of `act.act` is batch dimension and
    /// `act.act.size()[1..]` is equal to S::shape().
    ///
    /// TODO: explain action representation for the vectorized environment.
    fn filt(&mut self, act: PyGymEnvContinuousAct<S>) -> (PyObject, Record) {
        let act = act.act;
        let record =
            Record::from_slice(&[("act", RecordValue::Array1(act.iter().cloned().collect()))]);

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
            } else {
                // Interpret the first axis as processes in vectorized environments
                pyo3::Python::with_gil(|py| {
                    act.axis_iter(Axis(0))
                        .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
                        .collect::<Vec<_>>()
                        .into_py(py)
                })
            }
        };
        (act, record)
    }
}

// /// Convert [ArrayD<f32>] to [PyObject].
// ///
// /// The first element of the shape of `act` is batch dimension and
// /// `act.size()[1..]` is equal to S::shape().
// ///
// /// TODO: explain how to handle the first dimension for vectorized environment.
// pub fn to_pyobj<S: Shape>(act: ArrayD<f32>) -> PyObject {
//     if S::squeeze_first_dim() {
//         debug_assert_eq!(act.shape()[0], 1);
//         debug_assert_eq!(&act.shape()[1..], S::shape());
//         let act = act.remove_axis(ndarray::Axis(0));
//         pyo3::Python::with_gil(|py| {
//             let act = PyArrayDyn::<f32>::from_array(py, &act);
//             act.into_py(py)
//         })
//     } else {
//         // Interpret the first axis as processes in vectorized environments
//         pyo3::Python::with_gil(|py| {
//             act.axis_iter(Axis(0))
//                 .map(|act| PyArrayDyn::<f32>::from_array(py, &act))
//                 .collect::<Vec<_>>()
//                 .into_py(py)
//         })
//     }
// }

#[macro_export]
macro_rules! newtype_act_c {
    ($struct_:ident) => {
        #[derive(Clone, Debug)]
        struct $struct_(border_py_gym_env::PyGymEnvContinuousAct);

        impl $struct_ {
            fn new(act: ArrayD<f32>) -> Self {
                $struct_(border_py_gym_env::PyGymEnvContinuousAct::new(act))
            }
        }

        impl border_core::Act for $struct_ {}
    };
    ($struct_:ident, $struct2_:ident) => {
        newtype_act_c!($struct_);

        struct $struct2_(border_py_gym_env::PyGymEnvContinuousActRawFilter);

        impl border_py_gym_env::PyGymEnvActFilter<$struct_> for $struct2_ {
            fn filt(&mut self, act: $struct_) -> (pyo3::PyObject, border_core::record::Record) {
                self.0.filt(act.0)
            }
        }

        impl std::default::Default for $struct2_ {
            fn default() -> Self {
                Self(border_py_gym_env::PyGymEnvContinuousActRawFilter::default())
            }
        }
    };
}