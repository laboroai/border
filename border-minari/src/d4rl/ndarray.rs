use std::fmt::Debug;

use crate::{
    util::ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    MinariConverter,
};
use anyhow::Result;
use border_core::generic_replay_buffer::BatchBase;
use ndarray::{s, ArrayD, Axis, IxDyn, Slice};
use pyo3::{PyAny, PyObject};

/// State of the Kitchen environment represented by ndarray.
#[derive(Clone, Debug)]
pub struct KitchenState {
    pub kettle: ArrayD<f32>,
}

impl KitchenState {
    pub fn new(capacity: usize) -> Self {
        Self {
            kettle: ArrayD::zeros(IxDyn(&[capacity, 7])),
        }
    }

    pub fn push(&mut self, ix: usize, data: Self) {
        self.kettle.slice_mut(s![ix.., ..]).assign(&data.kettle);
    }

    pub fn sample(&self, ixs: &Vec<usize>) -> Self {
        Self {
            kettle: self.kettle.select(Axis(0), ixs),
        }
    }
}

/// Observation of the Kitchen environment represented by ndarray.
#[derive(Clone, Debug)]
pub struct KitchenObs {
    pub achieved_goal: KitchenState,
    pub desired_goal: KitchenState,
}

impl border_core::Obs for KitchenObs {
    fn len(&self) -> usize {
        self.achieved_goal.kettle.shape()[0]
    }
}

/// Batch of observation.
#[derive(Debug)]
pub struct KitchenObsBatch {
    pub achieved_goal: KitchenState,
    pub desired_goal: KitchenState,
}

impl BatchBase for KitchenObsBatch {
    fn new(capacity: usize) -> Self {
        Self {
            achieved_goal: KitchenState::new(capacity),
            desired_goal: KitchenState::new(capacity),
        }
    }

    fn push(&mut self, ix: usize, data: Self) {
        self.achieved_goal.push(ix, data.achieved_goal);
        self.desired_goal.push(ix, data.desired_goal);
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        Self {
            achieved_goal: self.achieved_goal.sample(ixs),
            desired_goal: self.desired_goal.sample(ixs),
        }
    }
}

impl From<KitchenObs> for KitchenObsBatch {
    fn from(obs: KitchenObs) -> Self {
        Self {
            achieved_goal: obs.achieved_goal,
            desired_goal: obs.desired_goal,
        }
    }
}

/// Action of the Kitchen environment represented by ndarray.
#[derive(Clone, Debug)]
pub struct KitchenAct {
    pub action: ArrayD<f32>,
}

impl border_core::Act for KitchenAct {}

/// Batch of action.
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

// Converter.
pub struct KitchenNdarrayConverter {}

impl MinariConverter for KitchenNdarrayConverter {
    type Obs = KitchenObs;
    type Act = KitchenAct;
    type ObsBatch = KitchenObsBatch;
    type ActBatch = KitchenActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let achieved_goal = obj.get_item("achieved_goal")?;
        let desired_goal = obj.get_item("desired_goal")?;
        Ok(KitchenObs {
            achieved_goal: KitchenState {
                kettle: pyobj_to_arrayd::<f64, f32>(achieved_goal.get_item("kettle")?.extract()?),
            },
            desired_goal: KitchenState {
                kettle: pyobj_to_arrayd::<f64, f32>(desired_goal.get_item("kettle")?.extract()?),
            },
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(act.action))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let achieved_goal = obj.get_item("achieved_goal")?;
        let desired_goal = obj.get_item("desired_goal")?;
        Ok(KitchenObsBatch {
            achieved_goal: KitchenState {
                kettle: pyobj_to_arrayd1(achieved_goal, "kettle")?,
            },
            desired_goal: KitchenState {
                kettle: pyobj_to_arrayd1(desired_goal, "kettle")?,
            },
        })
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let achieved_goal = obj.get_item("achieved_goal")?;
        let desired_goal = obj.get_item("desired_goal")?;

        // // Check the keys in achieved_goal
        // println!("{:?}", achieved_goal.call_method0("keys")?);
        // panic!();

        Ok(KitchenObsBatch {
            achieved_goal: KitchenState {
                kettle: pyobj_to_arrayd2(achieved_goal, "kettle")?,
            },
            desired_goal: KitchenState {
                kettle: pyobj_to_arrayd2(desired_goal, "kettle")?,
            },
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
