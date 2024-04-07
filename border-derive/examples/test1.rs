use border_derive::{Act, SubBatch};
use border_py_gym_env::GymDiscreteAct;
use border_tch_agent::TensorSubBatch;
use ndarray::ArrayD;
use std::convert::TryFrom;
use tch::Tensor;

#[derive(Debug, Clone)]
struct Obs(ArrayD<f32>);

#[derive(SubBatch)]
struct ObsBatch(TensorSubBatch);

impl From<Obs> for Tensor {
    fn from(value: Obs) -> Self {
        Tensor::try_from(&value.0).unwrap()
    }
}

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(Clone, Debug, Act)]
struct Act(GymDiscreteAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

fn main() {}
