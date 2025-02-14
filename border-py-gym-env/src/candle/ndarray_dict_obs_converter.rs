//! Converter for observation and action of [`NdarrayDictObs`] and [`NdarrayAct`].
use super::{NdarrayAct, NdarrayDictObs};
use crate::{util::pyobj_to_arrayd, GymEnvConverter};
use anyhow::Result;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Configuration of [`NdarrayConverter`].
pub struct NdarrayDictObsConverterConfig {
    pub keys: Vec<String>,
}

impl Default for NdarrayDictObsConverterConfig {
    fn default() -> Self {
        Self { keys: vec![] }
    }
}

#[derive(Clone, Debug)]
/// Converter for observation and action of [`NdarrayDictObs`] and [`NdarrayAct`].
///
/// This struct supports continuous and discrete actions.
/// The former is represented as a vector, while the latter is represented as an integer.
/// The action type is automatically detected from samples, those are outputs
/// of the model being trained.
pub struct NdarrayDictObsConverter {
    obs_keys: Vec<String>,
}

impl GymEnvConverter for NdarrayDictObsConverter {
    type Obs = NdarrayDictObs;
    type Act = NdarrayAct;
    type Config = NdarrayDictObsConverterConfig;

    fn new(config: &Self::Config) -> Result<Self> {
        let converter = Self {
            obs_keys: config.keys.clone(),
        };
        Ok(converter)
    }

    /// Convert [`PyObject`] to [`NdarrayDictObs`].
    fn filt_obs(&mut self, obs: PyObject) -> Result<Self::Obs> {
        let obs = pyo3::Python::with_gil(|py| {
            self.obs_keys
                .iter()
                .map(|key| {
                    let pyobj = obs.call_method1(py, "get", (key,)).unwrap();
                    (key.clone(), pyobj_to_arrayd::<f32, f32>(pyobj))
                })
                .collect::<Vec<_>>()
        });
        Ok(NdarrayDictObs(obs))
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
