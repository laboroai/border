//! Observation, action types and corresponding converters for the AntMaze environment implemented with ndarray.
use std::fmt::Debug;

use crate::{
    util::ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    MinariConverter,
};
use anyhow::Result;
use border_core::generic_replay_buffer::BatchBase;
use ndarray::{s, ArrayD, Axis, IxDyn, Slice};
use pyo3::{PyAny, PyObject, Python};

const DIM_OBS: usize = 27;
const DIM_ACT: usize = 8;

/// Observation of the AntMaze environment stored as ndarray.
///
/// It contains a 27-dimensional vector as explained
/// [here](https://robotics.farama.org/envs/maze/ant_maze/).
///
/// Since the observation of the environment is coming from Python interpreter, this struct
/// can be converted from [`PyObject`].
///
/// To create of batch of observations, this struct can be converted into [`AntMazeObsBatch`].
#[derive(Clone, Debug)]
pub struct AntMazeObs {
    pub obs: ArrayD<f32>,
}

impl border_core::Obs for AntMazeObs {
    fn len(&self) -> usize {
        self.obs.shape()[0]
    }
}

/// Batch of observations.
///
/// It can be converted from an observation, i.e., instance of [`AntMazeObs`].
///
/// It can be converted into an ndarray.
#[derive(Debug)]
pub struct AntMazeObsBatch {
    pub obs: ArrayD<f32>,
}

impl BatchBase for AntMazeObsBatch {
    fn new(capacity: usize) -> Self {
        Self {
            obs: ArrayD::zeros(IxDyn(&[capacity, DIM_OBS])),
        }
    }

    fn push(&mut self, ix: usize, data: Self) {
        self.obs.slice_mut(s![ix.., ..]).assign(&data.obs);
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        Self {
            obs: self.obs.select(Axis(0), ixs),
        }
    }
}

impl From<AntMazeObs> for AntMazeObsBatch {
    fn from(obs: AntMazeObs) -> Self {
        Self { obs: obs.obs }
    }
}

/// Action of the AntMaze environment stored as ndarray.
///
/// To create a batch of actions, this struct can be converted into [`AntMazeActBatch`].
#[derive(Clone, Debug)]
pub struct AntMazeAct {
    pub action: ArrayD<f32>,
}

impl border_core::Act for AntMazeAct {}

/// Batch of actions.
#[derive(Debug)]
pub struct AntMazeActBatch {
    pub action: ArrayD<f32>,
}

impl AntMazeActBatch {
    /// Returns an action at the specified index in the batch.
    pub fn get(&self, ix: usize) -> AntMazeAct {
        AntMazeAct {
            action: self.action.select(Axis(0), &[ix]).to_owned(),
        }
    }
}

impl BatchBase for AntMazeActBatch {
    fn new(capacity: usize) -> Self {
        Self {
            action: ArrayD::zeros(IxDyn(&[capacity, DIM_ACT])),
        }
    }

    fn push(&mut self, ix: usize, data: Self) {
        self.action.slice_mut(s![ix.., ..]).assign(&data.action);
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        Self {
            action: self.action.select(Axis(0), ixs),
        }
    }
}

impl From<AntMazeAct> for AntMazeActBatch {
    fn from(act: AntMazeAct) -> Self {
        Self { action: act.action }
    }
}

/// Converter for the AntMaze environment implemented with ndarray.
pub struct AntMazeConverter {}

impl MinariConverter for AntMazeConverter {
    type Obs = AntMazeObs;
    type Act = AntMazeAct;
    type ObsBatch = AntMazeObsBatch;
    type ActBatch = AntMazeActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let obs = obj.get_item("observation")?;
        Ok(AntMazeObs {
            obs: pyobj_to_arrayd::<f64, f32>(obs.into()),
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(act.action))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(AntMazeObsBatch {
            obs: pyobj_to_arrayd1(obj, "observation")?,
        })
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(AntMazeObsBatch {
            obs: pyobj_to_arrayd2(obj, "observation")?,
        })
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(AntMazeActBatch {
            action: { pyobj_to_arrayd::<f32, f32>(obj.into()) },
        })
    }

    fn env_params(&self, _py: Python<'_>) -> Vec<(&str, PyObject)> {
        vec![]
    }
}

/// Converts PyObject to ArrayD and drop the last row.
fn pyobj_to_arrayd1(obj: &PyAny, name: &str) -> Result<ArrayD<f32>> {
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);
    Ok(arr.slice_axis(Axis(0), Slice::from(..-1)).to_owned())
}

/// Converts PyObject to ArrayD and drop the first row.
fn pyobj_to_arrayd2(obj: &PyAny, name: &str) -> Result<ArrayD<f32>> {
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);
    Ok(arr.slice_axis(Axis(0), Slice::from(1..)).to_owned())
}
