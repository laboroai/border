//! Observation, actiontypes and corresponding converters for the Kitchen environment impmented with ndarray.
use std::fmt::Debug;

use crate::{
    util::ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    MinariConverter,
};
use anyhow::Result;
use border_core::generic_replay_buffer::BatchBase;
use ndarray::{s, ArrayD, Axis, IxDyn, Slice};
use pyo3::{PyAny, PyObject};

/// Observation of the Kitchen environment stored as ndarray.
///
/// It contains a 59-dimensional vector as explained
/// [here](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/#observation-space).
///
/// Since the observation of the environment is coming from Python interpreter, this struct
/// can be converted from [`PyObject`].
///
/// To create of batch of observations, this struct can be converted into [`KitchenObsBatch`].
#[derive(Clone, Debug)]
pub struct KitchenObs {
    pub obs: ArrayD<f32>,
}

impl border_core::Obs for KitchenObs {
    fn dummy(n: usize) -> Self {
        Self {
            obs: ArrayD::zeros(IxDyn(&[n, 59])),
        }
    }

    fn len(&self) -> usize {
        self.obs.shape()[0]
    }
}

/// Batch of observations.
///
/// It can be converted from an observation, i.e., instance of [`KitchenObs`].
///
/// It can be converted into an ndarray.
#[derive(Debug)]
pub struct KitchenObsBatch {
    pub obs: ArrayD<f32>,
}

impl BatchBase for KitchenObsBatch {
    fn new(capacity: usize) -> Self {
        Self {
            obs: ArrayD::zeros(IxDyn(&[capacity, 59])),
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

impl From<KitchenObs> for KitchenObsBatch {
    fn from(obs: KitchenObs) -> Self {
        Self { obs: obs.obs }
    }
}

/// Action of the Kitchen environment stored as ndarray.
///
/// To create a batch of actions, this struct can be converted into [`KitchenActBatch`].
#[derive(Clone, Debug)]
pub struct KitchenAct {
    pub action: ArrayD<f32>,
}

impl border_core::Act for KitchenAct {}

/// Batch of actions.
#[derive(Debug)]
pub struct KitchenActBatch {
    pub action: ArrayD<f32>,
}

impl KitchenActBatch {
    /// Returns an action at the specified index in the batch.
    pub fn get(&self, ix: usize) -> KitchenAct {
        KitchenAct {
            action: self.action.select(Axis(0), &[ix]).to_owned(),
        }
    }
}

impl BatchBase for KitchenActBatch {
    fn new(capacity: usize) -> Self {
        Self {
            action: ArrayD::zeros(IxDyn(&[capacity, 9])),
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

impl From<KitchenAct> for KitchenActBatch {
    fn from(act: KitchenAct) -> Self {
        Self { action: act.action }
    }
}

/// Converter for the Kitchen environment implemented with ndarray.
pub struct KitchenConverter {}

impl MinariConverter for KitchenConverter {
    type Obs = KitchenObs;
    type Act = KitchenAct;
    type ObsBatch = KitchenObsBatch;
    type ActBatch = KitchenActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let obs = obj.get_item("observation")?;
        Ok(KitchenObs {
            obs: pyobj_to_arrayd::<f64, f32>(obs.into()),
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(act.action))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(KitchenObsBatch {
            obs: pyobj_to_arrayd1(obj, "observation")?,
        })
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(KitchenObsBatch {
            obs: pyobj_to_arrayd2(obj, "observation")?,
        })
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(KitchenActBatch {
            action: { pyobj_to_arrayd::<f64, f32>(obj.into()) },
        })
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
