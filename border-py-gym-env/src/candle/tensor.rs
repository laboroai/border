//! Converter for Actervation and action of [`candle_core::Tensor`].
use super::{arrayd_to_tensor, tensor_to_arrayd, TensorBatch};
use crate::{
    util::{arrayd_to_pyobj, pyobj_to_arrayd, ActionType},
    GymEnvConverter,
};
use anyhow::Result;
use candle_core::Tensor;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};
use serde::{Deserialize, Serialize};

pub use act::*;
pub use converter::*;
pub use obs::*;

mod obs {
    use super::*;

    #[derive(Clone, Debug)]
    /// Observation.
    pub struct TensorObs(pub(super) Tensor);

    impl border_core::Obs for TensorObs {
        fn len(&self) -> usize {
            1
        }
    }

    impl Into<Tensor> for TensorObs {
        fn into(self) -> Tensor {
            self.0
        }
    }

    impl From<TensorObs> for TensorBatch {
        fn from(o: TensorObs) -> Self {
            o.0.into()
        }
    }
}

mod act {
    use super::*;

    #[derive(Clone, Debug)]
    /// Action.
    pub struct TensorAct(pub(super) Tensor);

    impl border_core::Act for TensorAct {}

    impl Into<Tensor> for TensorAct {
        fn into(self) -> Tensor {
            self.0
        }
    }

    impl From<Tensor> for TensorAct {
        fn from(t: Tensor) -> Self {
            Self(t)
        }
    }

    impl From<TensorAct> for TensorBatch {
        fn from(a: TensorAct) -> Self {
            a.0.into()
        }
    }
}

mod converter {
    use super::*;

    #[derive(Clone, Debug, Deserialize, Serialize)]
    /// Configuration of [`TensorConverter`].
    pub struct TensorConverterConfig {
        pub action_type: ActionType,
    }

    impl Default for TensorConverterConfig {
        fn default() -> Self {
            Self {
                action_type: ActionType::Discrete,
            }
        }
    }

    #[derive(Clone, Debug)]
    /// Converter for observation and action of [`candle_core::Tensor`].
    pub struct TensorConverter {
        action_type: ActionType,
    }

    impl GymEnvConverter for TensorConverter {
        type Obs = TensorObs;
        type Act = TensorAct;
        type Config = TensorConverterConfig;

        fn new(config: &Self::Config) -> Result<Self> {
            let converter = Self {
                action_type: config.action_type.clone(),
            };
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

            // ndarray to tensor
            let tensor = arrayd_to_tensor::<f32, f32>(obs, false)?;

            Ok(TensorObs(tensor))
        }

        /// Convert [`Self::Act`] to [`PyObject`].
        fn filt_act(&mut self, act: Self::Act) -> Result<PyObject> {
            match self.action_type {
                ActionType::Discrete => {
                    // let vec = act
                    //     .0
                    //     .to_vec1::<i64>()
                    //     .expect("Failed to convert Tensor to Act");
                    let arrayd = tensor_to_arrayd::<i64>(act.0, true)?;
                    let pyobj = pyo3::Python::with_gil(|py| {
                        let act = PyArrayDyn::<i64>::from_array(py, &arrayd);
                        act.into_py(py)
                    });
                    Ok(pyobj)
                }
                ActionType::Continuous => {
                    let arrayd = tensor_to_arrayd::<f32>(act.0, true)?;
                    Ok(arrayd_to_pyobj(arrayd))
                }
            }
        }
    }
}
