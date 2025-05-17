//! Observation, actiontypes and corresponding converters for the Point Maze environment implemented with ndarray.
use std::fmt::Debug;

use crate::{
    util::ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    MinariConverter,
};
use anyhow::Result;
use border_core::generic_replay_buffer::BatchBase;
use ndarray::{s, ArrayD, Axis, IxDyn, Slice};
use pyo3::{PyAny, PyObject, Python};

const DIM_OBS: usize = 4;
const DIM_ACT: usize = 2;

/// Observation of the Point Maze environment stored as ndarray.
///
/// It contains a 27-dimensional vector as explained
/// [here](https://robotics.farama.org/envs/maze/point_maze/).
///
/// Since the observation of the environment is coming from Python interpreter, this struct
/// can be converted from [`PyObject`].
///
/// To create of batch of observations, this struct can be converted into [`PointMazeObsBatch`].
#[derive(Clone, Debug)]
pub struct PointMazeObs {
    pub obs: ArrayD<f32>,
}

impl border_core::Obs for PointMazeObs {
    fn len(&self) -> usize {
        self.obs.shape()[0]
    }
}

/// Batch of observations.
///
/// It can be converted from an observation, i.e., instance of [`PointMazeObs`].
///
/// It can be converted into an ndarray.
#[derive(Clone, Debug)]
pub struct PointMazeObsBatch {
    pub obs: ArrayD<f32>,
}

impl BatchBase for PointMazeObsBatch {
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

impl From<PointMazeObs> for PointMazeObsBatch {
    fn from(obs: PointMazeObs) -> Self {
        Self { obs: obs.obs }
    }
}

/// Action of the PointMaze environment stored as ndarray.
///
/// To create a batch of actions, this struct can be converted into [`PointMazeActBatch`].
#[derive(Clone, Debug)]
pub struct PointMazeAct {
    pub action: ArrayD<f32>,
}

impl border_core::Act for PointMazeAct {}

/// Batch of actions.
#[derive(Clone, Debug)]
pub struct PointMazeActBatch {
    pub action: ArrayD<f32>,
}

impl PointMazeActBatch {
    /// Returns an action at the specified index in the batch.
    pub fn get(&self, ix: usize) -> PointMazeAct {
        PointMazeAct {
            action: self.action.select(Axis(0), &[ix]).to_owned(),
        }
    }
}

impl BatchBase for PointMazeActBatch {
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

impl From<PointMazeAct> for PointMazeActBatch {
    fn from(act: PointMazeAct) -> Self {
        Self { action: act.action }
    }
}

/// Converter for the Point Maze environment implemented with ndarray.
pub struct PointMazeConverter {}

impl MinariConverter for PointMazeConverter {
    type Obs = PointMazeObs;
    type Act = PointMazeAct;
    type ObsBatch = PointMazeObsBatch;
    type ActBatch = PointMazeActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let obs = obj.get_item("observation")?;
        Ok(PointMazeObs {
            obs: pyobj_to_arrayd::<f64, f32>(obs.into()),
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(act.action))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(PointMazeObsBatch {
            obs: pyobj_to_arrayd1(obj, "observation")?,
        })
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(PointMazeObsBatch {
            obs: pyobj_to_arrayd2(obj, "observation")?,
        })
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(PointMazeActBatch {
            action: { pyobj_to_arrayd::<f32, f32>(obj.into()) },
        })
    }

    fn env_params(&self, _py: Python<'_>) -> Vec<(&str, PyObject)> {
        // not override the original parameters in Minari
        // https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/cc54b71147650b310f5a84c642dd6dc127f333a1/scripts/pointmaze/create_pointmaze_dataset.py#L157-L159
        vec![]

        // When want to override the parameters, comment in the following code:
        // vec![
        //     ("continuing_task", Some("True")),
        //     ("reset_target", Some("True")),
        // ]
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
