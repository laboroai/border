use border_derive::{Obs, SubBatch, Act};
use border_py_gym_env::{PyGymEnvObs, PyGymEnvDiscreteAct};
use border_core::shape;
use std::convert::TryFrom;
use border_tch_agent::TensorSubBatch;

shape!(ObsShape, [4]);
shape!(ActShape, [1]);

#[derive(Debug, Clone, Obs)]
struct Obs(PyGymEnvObs<ObsShape, f64, f32>);

#[derive(SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, f32>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvDiscreteAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, i64>);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

fn main() {}
