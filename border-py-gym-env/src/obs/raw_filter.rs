use super::{pyobj_to_arrayd, PyGymEnvObs};
use crate::PyGymEnvObsFilter;
use border_core::{
    record::{Record, RecordValue},
    Obs, Shape,
};
use num_traits::cast::AsPrimitive;
use numpy::Element;
use pyo3::PyObject;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [PyGymEnvObsRawFilter].
#[derive(Clone)]
pub struct PyGymEnvObsRawFilterConfig {
    vectorized: bool,
}

impl Default for PyGymEnvObsRawFilterConfig {
    fn default() -> Self {
        Self { vectorized: false }
    }
}

/// An observation filter without any postprocessing.
///
/// The filter works with [PyGymEnv](crate::PyGymEnv).
pub struct PyGymEnvObsRawFilter<S, T1, T2, U> {
    /// If the environment is vectorized.
    pub vectorized: bool,
    /// Marker.
    pub phantom: PhantomData<(S, T1, T2, U)>,
}

impl<S, T1, T2, U> Default for PyGymEnvObsRawFilter<S, T1, T2, U> {
    fn default() -> Self {
        Self {
            vectorized: false,
            phantom: PhantomData,
        }
    }
}

impl<S, T1, T2, U> PyGymEnvObsFilter<U> for PyGymEnvObsRawFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + Debug + num_traits::Zero + AsPrimitive<f32>,
    U: Obs + From<PyGymEnvObs<S, T1, T2>>,
{
    type Config = PyGymEnvObsRawFilterConfig;

    fn build(config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            vectorized: config.vectorized,
            phantom: PhantomData,
        })
    }

    /// Convert `PyObject` to [ndarray::ArrayD].
    ///
    /// No filter is applied after conversion.
    /// The shape of the observation is `S` in [PyGymEnv](crate::PyGymEnv).
    ///
    /// For [crate::PyVecGymEnv], which is a vectorized environments,
    /// the shape becomes `[n_procs, S]`, where `n_procs` is the number of processes
    /// of the vectorized environment.
    ///
    /// [Record] in the returned value has `obs`, which is a flattened array of
    /// observation, for either of single and vectorized environments.
    fn filt(&mut self, obs: PyObject) -> (U, Record) {
        if self.vectorized {
            unimplemented!();
            // let obs = pyo3::Python::with_gil(|py| {
            //     debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");
            //     let obs: Py<PyList> = obs.extract(py).unwrap();

            //     // Iterate over the list of observations of the environments in the
            //     // vectorized environment.
            //     let filtered = obs
            //         .as_ref(py)
            //         .iter()
            //         .map(|o| {
            //             // `NoneType` means the element will be ignored in the following processes.
            //             // This can appears in partial reset of the vectorized environment.
            //             if o.get_type().name().unwrap() == "NoneType" {
            //                 ArrayD::zeros(IxDyn(S::shape()))
            //             }
            //             // Processes the partial observation in the vectorized environment.
            //             else {
            //                 debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
            //                 let obs: &PyArrayDyn<T1> = o.extract().unwrap();
            //                 let obs = obs.to_owned_array();
            //                 let obs = obs.mapv(|elem| elem.as_());
            //                 debug_assert_eq!(obs.shape(), S::shape());
            //                 obs
            //             }
            //         })
            //         .collect::<Vec<_>>();
            //     let arrays_view: Vec<_> = filtered.iter().map(|a| a.view()).collect();
            //     PyGymEnvObs::<S, T1, T2> {
            //         obs: stack(Axis(0), arrays_view.as_slice()).unwrap(),
            //         phantom: PhantomData,
            //     }
            // });
            // let record = {
            //     let vec = obs.obs.iter().map(|x| x.as_()).collect();
            //     Record::from_slice(&[("obs", RecordValue::Array1(vec))])
            // };
            // // let record = Record::from_slice(
            // //     &[("obs", RecordValue::Array1(
            // //         Vec::<_>::from_iter(obs.obs.iter().map(|x| x.as_()).cloned()))
            // //     )]);
            // (obs, record)
        } else {
            let obs = pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() == "NoneType" {
                    // TODO: consider panic!() if the environment returns None
                    // PyGymEnvObs::<S, T1, T2>::dummy(1)
                    panic!();
                } else {
                    PyGymEnvObs {
                        obs: pyobj_to_arrayd::<S, T1, T2>(obs),
                        phantom: PhantomData,
                    }
                }
            });
            // let array1: Vec<T2> = obs.obs.iter().cloned().collect();
            // let record = Record::from_slice(&[("obs", RecordValue::Array1(array1))]);
            let record = {
                // let vec = Vec::<f32>::from_iter(obs.obs.iter().map(|x| x.as_()));
                let vec = obs.obs.iter().map(|x| x.as_()).collect();
                Record::from_slice(&[("obs", RecordValue::Array1(vec))])
            };
            (obs.into(), record)
        }
    }
}
