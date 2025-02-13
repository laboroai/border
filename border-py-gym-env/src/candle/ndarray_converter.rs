//! Converter for Observation and action of [`ArrayD`].
use super::{NdarrayAct, NdarrayObs};
use crate::{util::pyobj_to_arrayd, GymEnvConverter};
use anyhow::Result;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Configuration of [`NdarrayConverter`].
pub struct NdarrayConverterConfig {}

impl Default for NdarrayConverterConfig {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug)]
/// Converter for observation and action of [`ArrayD`].
///
/// This struct supports continuous and discrete actions.
/// The former is represented as a vector, while the latter is represented as an integer.
/// The action type is automatically detected from samples, those are outputs
/// of the model being trained.
pub struct NdarrayConverter {}

impl GymEnvConverter for NdarrayConverter {
    type Obs = NdarrayObs;
    type Act = NdarrayAct;
    type Config = NdarrayConverterConfig;

    fn new(_config: &Self::Config) -> Result<Self> {
        let converter = Self {};
        Ok(converter)
    }

    /// Convert observation.
    ///
    /// Data type should be f32.
    fn filt_obs(&mut self, obs: PyObject) -> Result<Self::Obs> {
        // ndarray
        let obs = pyo3::Python::with_gil(|py| {
            if obs.as_ref(py).get_type().name().unwrap() == "NoneType" {
                panic!();
            } else {
                pyobj_to_arrayd::<f32, f32>(obs)
            }
        });

        Ok(NdarrayObs(obs))
    }

    /// Convert [`Self::Act`] to [`PyObject`].
    fn filt_act(&mut self, act: Self::Act) -> Result<PyObject> {
        match act {
            NdarrayAct::Continuous(arrayd) => {
                let pyobj = pyo3::Python::with_gil(|py| {
                    let act = PyArrayDyn::<f32>::from_array(py, &arrayd);
                    act.into_py(py)
                });
                Ok(pyobj)
            }
            NdarrayAct::Discrete(arrayd) => {
                let pyobj = pyo3::Python::with_gil(|py| {
                    let act = PyArrayDyn::<i64>::from_array(py, &arrayd);
                    act.into_py(py)
                });
                Ok(pyobj)
            }
        }
    }
}
