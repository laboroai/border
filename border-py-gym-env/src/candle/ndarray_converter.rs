//! Converter for Observation and action of `ArrayD`.
use super::{arrayd_to_tensor, tensor_to_arrayd, TensorBatch};
use crate::{
    util::{arrayd_to_pyobj, pyobj_to_arrayd, ActionType},
    GymEnvConverter,
};
use anyhow::Result;
use candle_core::Tensor;
use ndarray::ArrayD;
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
    pub struct NdarrayObs(pub(super) ArrayD<f32>);

    impl border_core::Obs for NdarrayObs {
        fn len(&self) -> usize {
            self.0.shape()[0]
        }
    }

    impl Into<Tensor> for NdarrayObs {
        fn into(self) -> Tensor {
            arrayd_to_tensor::<_, f32>(self.0, false).unwrap()
        }
    }

    impl From<NdarrayObs> for TensorBatch {
        fn from(o: NdarrayObs) -> Self {
            TensorBatch::from_tensor(o.into())
        }
    }
}

mod act {
    use super::*;

    #[derive(Clone, Debug)]
    /// Action.
    pub struct NdarrayAct(pub(super) ArrayD<f32>);

    impl border_core::Act for NdarrayAct {}

    impl Into<Tensor> for NdarrayAct {
        fn into(self) -> Tensor {
            arrayd_to_tensor::<_, f32>(self.0, true).unwrap()
        }
    }

    impl From<Tensor> for NdarrayAct {
        fn from(t: Tensor) -> Self {
            Self(tensor_to_arrayd(t, true).unwrap())
        }
    }

    impl From<NdarrayAct> for TensorBatch {
        fn from(act: NdarrayAct) -> Self {
            let tensor = act.into();
            TensorBatch::from_tensor(tensor)
        }
    }
}

mod converter {
    use super::*;

    #[derive(Clone, Debug, Deserialize, Serialize)]
    /// Configuration of [`NdarrayConverter`].
    pub struct NdarrayConverterConfig {
        pub action_type: ActionType,
    }

    impl Default for NdarrayConverterConfig {
        fn default() -> Self {
            Self {
                action_type: ActionType::Discrete,
            }
        }
    }

    #[derive(Clone, Debug)]
    /// Converter for observation and action of [`candle_core::Tensor`].
    pub struct NdarrayConverter {
        action_type: ActionType,
    }

    impl GymEnvConverter for NdarrayConverter {
        type Obs = NdarrayObs;
        type Act = NdarrayAct;
        type Config = NdarrayConverterConfig;

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

            Ok(NdarrayObs(obs))
        }

        /// Convert [`Self::Act`] to [`PyObject`].
        fn filt_act(&mut self, act: Self::Act) -> Result<PyObject> {
            match self.action_type {
                ActionType::Discrete => {
                    panic!();
                    // let vec = act
                    //     .0
                    //     .to_vec1::<i64>()
                    //     .expect("Failed to convert Tensor to Act");
                    // let arrayd = tensor_to_arrayd::<i64>(act.0, true)?;
                    // let pyobj = pyo3::Python::with_gil(|py| {
                    //     let act = PyArrayDyn::<i64>::from_array(py, &arrayd);
                    //     act.into_py(py)
                    // });
                    // Ok(pyobj)
                }
                ActionType::Continuous => Ok(arrayd_to_pyobj(act.0)),
            }
        }
    }
}
