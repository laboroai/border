
use tch::{Tensor};
use crate::py_gym_env::{PyNDArrayObs, PyGymDiscreteAct};
use crate::agents::adapter::{TchObsAdapter, TchActAdapter};

pub struct PyNDArrayObsAdapter {
    shape: Vec<i64>
}

impl PyNDArrayObsAdapter {
    pub fn new(shape: &[i64]) -> Self {
        Self {
            shape: shape.into()
        }
    }
}

impl TchObsAdapter<PyNDArrayObs> for PyNDArrayObsAdapter {
    fn convert(&self, obs: &PyNDArrayObs) -> Tensor {
        let obs = obs.0.view().to_slice().unwrap();
        Tensor::of_slice(obs)
    }

    fn shape(&self) -> &[i64] {
        self.shape.as_slice()
    }
}

pub struct PyGymDiscreteActAdapter {
    shape: Vec<i64>
}

impl PyGymDiscreteActAdapter {
    pub fn new(shape: &[i64]) -> Self {
        Self {
            shape: shape.into()
        }
    }
}

impl TchActAdapter<PyGymDiscreteAct> for PyGymDiscreteActAdapter {
    fn convert(&self, act: &Tensor) -> PyGymDiscreteAct {
        let a: i32 = act.into();
        PyGymDiscreteAct::new(a as u32)
    }

    fn back(&self, act: &PyGymDiscreteAct) -> Tensor {
        (act.0 as i64).into()
    }

    fn shape(&self) -> &[i64] {
        self.shape.as_slice()
    }
}
