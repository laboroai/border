//! Observation for [`super::PyGymEnv`] and [`super::PyVecGymEnv`].
use std::{fmt::Debug, iter::FromIterator};
use std::default::Default;
use std::marker::PhantomData;
use log::trace;
use num_traits::cast::AsPrimitive;
use pyo3::{Py, PyObject, types::PyList};
use ndarray::{ArrayD, Axis, IxDyn, stack};
use numpy::{Element, PyArrayDyn};

use crate::{
    core::{Obs, record::{Record, RecordValue}},
    env::py_gym_env::{Shape, PyGymEnvObsFilter},
};

fn any(is_done: &[i8]) -> bool {
    is_done.iter().fold(0, |x, v| x + *v as i32) > 0
}

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

/// Observation represented by an [ndarray::ArrayD].
///
/// `S` is the shape of an observation, except for batch and process dimensions.
/// `T` is the dtype of ndarray in the Python gym environment.
/// For some reason, the dtype of observations in Python gym environments seems to 
/// vary, f32 or f64. To get observations in Rust side, the dtype is specified as a
/// type parameter, instead of checking the dtype of Python array at runtime.
#[derive(Clone, Debug)]
pub struct PyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug,
{
    pub(crate) obs: ArrayD<f32>,
    pub(crate) phantom: PhantomData<(S, T)>
}

impl<S, T> PyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug,
{
    /// Create an empty observation, used in pong module.
    pub fn new(obs: ArrayD<f32>) -> Self {
        Self {
            obs,
            phantom: PhantomData
        }
    }
}

impl<S, T> Obs for PyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero,
{
    fn dummy(n_procs: usize) -> Self {
        let shape = &mut S::shape().to_vec();
        shape.insert(0, n_procs as _);
        trace!("Shape of TchPyGymEnvObs: {:?}", shape);
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData
        }
    }

    fn merge(mut self, obs_reset: Self, is_done: &[i8]) -> Self {
        if any(is_done) {
            for (i, is_done_i) in is_done.iter().enumerate() {
                if *is_done_i != 0 {
                    self.obs.index_axis_mut(Axis(0), i)
                        .assign(&obs_reset.obs.index_axis(Axis(0), i));
                }
            }
        };
        self
    }
}

/// An observation filter without any postprocessing.
///
/// The filter works with [super::PyGymEnv] or [super::PyVecGymEnv].
pub struct PyGymEnvObsRawFilter<S, T> {
    pub vectorized: bool,
    pub phantom: PhantomData<(S, T)>
}

impl<S, T> PyGymEnvObsRawFilter<S, T> {
    pub fn vectorized() -> Self {
        Self {
            vectorized: true,
            phantom: PhantomData,
        }
    }
}

impl<S, T> Default for PyGymEnvObsRawFilter<S, T> where
    S: Shape,
    T: Element,
{
    fn default() -> Self {
        Self {
            vectorized: false,
            phantom: PhantomData,
        }
    }
}

impl<S, T> PyGymEnvObsFilter<PyGymEnvObs<S, T>> for PyGymEnvObsRawFilter<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero + AsPrimitive<f32>,
{
    /// Convert `PyObject` to [ndarray::ArrayD].
    ///
    /// No filter is applied after conversion.
    /// The shape of the observation is `S` in [crate::env::py_gym_env::PyGymEnv].
    ///
    /// For [crate::env::py_gym_env::PyVecGymEnv], which is a vectorized environments,
    /// the shape becomes `[n_procs, S]`, where `n_procs` is the number of processes
    /// of the vectorized environment.
    ///
    /// [Record] in the returned value has `obs`, which is a flattened array of
    /// observation, for either of single and vectorized environments.
    fn filt(&mut self, obs: PyObject) -> (PyGymEnvObs<S, T>, Record) {
        if self.vectorized {
            let obs = pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");
                let obs: Py<PyList> = obs.extract(py).unwrap();

                // Iterate over the list of observations of the environments in the
                // vectorized environment.
                let filtered = obs.as_ref(py).iter()
                    .map(|o| {
                        // `NoneType` means the element will be ignored in the following processes.
                        // This can appears in partial reset of the vectorized environment.
                        if o.get_type().name().unwrap() == "NoneType" {
                            ArrayD::zeros(IxDyn(S::shape()))
                        }
                        // Processes the partial observation in the vectorized environment.
                        else {
                            debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
                            let obs: &PyArrayDyn<T> = o.extract().unwrap();
                            let obs = obs.to_owned_array();
                            let obs = obs.mapv(|elem| elem.as_());
                            debug_assert_eq!(obs.shape(), S::shape());
                            obs
                        }
                    }).collect::<Vec<_>>();
                let arrays_view: Vec<_> = filtered.iter().map(|a| a.view()).collect();
                PyGymEnvObs::<S, T> {
                    obs: stack(Axis(0), arrays_view.as_slice()).unwrap(),
                    phantom: PhantomData
                }
            });
            let record = Record::from_slice(
                &[("obs", RecordValue::Array1(Vec::<_>::from_iter(obs.obs.iter().cloned())))]);
            (obs, record)
        }
        else {
            let obs = pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() == "NoneType" {
                    // TODO: consider panic!() if the environment returns None
                    PyGymEnvObs::<S, T>::dummy(1)
                }
                else {
                    PyGymEnvObs {
                        obs: pyobj_to_arrayd::<S, T>(obs),
                        phantom: PhantomData,
                    }
                }
            });
            let array1: Vec<f32> = obs.obs.iter().cloned().collect();
            let record = Record::from_slice(&[("obs", RecordValue::Array1(array1))]);
            (obs, record)
        }
    }
}
