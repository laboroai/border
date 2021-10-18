use border_derive::{Obs, SubBatch, Act};
use border_py_gym_env::{PyGymEnvObs, PyGymEnvDiscreteAct};
use border_core::shape;
use tch::Tensor;
use std::convert::TryFrom;
use border_tch_agent::TensorSubBatch;

shape!(ObsShape, [4]);
shape!(ActShape, [1]);

#[derive(Debug, Clone, Obs)]
struct Obs(PyGymEnvObs<ObsShape, f64, f32>);

// impl From<Obs> for Tensor {
//     fn from(obs: Obs) -> Tensor {
//         Tensor::try_from(&obs.0.obs).unwrap()
//     }
// }

#[derive(SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, f32>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

impl From<ObsBatch> for Tensor {
    fn from(b: ObsBatch) -> Self {
        b.0.into()
    }
}

#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvDiscreteAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, i64>);

impl From<Act> for Tensor {
    fn from(act: Act) -> Tensor {
        let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();

        // The first dimension of the action tensor is the number of processes,
        // which is 1 for the non-vectorized environment.
        t.unsqueeze(0)
    }
}

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

impl From<ActBatch> for Tensor {
    fn from(act: ActBatch) -> Self {
        act.0.into()
    }
}

impl From<Tensor> for Act {
    // `t` must be a 1-dimentional tensor of `f32` (?)
    fn from(t: Tensor) -> Self {
        let data: Vec<i64> = t.into();
        let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
        Act(PyGymEnvDiscreteAct::new(data))
    }
}

fn main() {}
