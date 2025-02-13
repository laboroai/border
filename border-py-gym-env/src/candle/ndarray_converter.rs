//! Converter for Observation and action of `ArrayD`.
use super::{arrayd_to_tensor, tensor_to_arrayd, TensorBatch};
use crate::{util::pyobj_to_arrayd, GymEnvConverter};
use anyhow::Result;
use candle_core::{DType, Tensor, D};
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
    pub struct NdarrayObs(pub ArrayD<f32>);

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
    pub enum NdarrayAct {
        Continuous(ArrayD<f32>),
        Discrete(ArrayD<i64>),
    }

    impl border_core::Act for NdarrayAct {}

    impl Into<Tensor> for NdarrayAct {
        fn into(self) -> Tensor {
            match self {
                Self::Continuous(array) => arrayd_to_tensor::<_, f32>(array, true).unwrap(),
                Self::Discrete(array) => {
                    let t = arrayd_to_tensor::<_, i64>(array, true).unwrap();
                    t.unsqueeze(D::Minus1).unwrap()
                }
            }
        }
    }

    impl From<Tensor> for NdarrayAct {
        fn from(t: Tensor) -> Self {
            match t.dtype() {
                DType::F32 => Self::Continuous(tensor_to_arrayd(t, true).unwrap()),
                DType::I64 => Self::Discrete(tensor_to_arrayd(t, true).unwrap()),
                _ => panic!(),
            }
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
    pub struct NdarrayConverterConfig {}

    impl Default for NdarrayConverterConfig {
        fn default() -> Self {
            Self {}
        }
    }

    #[derive(Clone, Debug)]
    /// Converter for observation and action of [`candle_core::Tensor`].
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
}
