use std::{
    convert::{TryFrom, TryInto},
    fmt::Debug,
};

use crate::{
    util::ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    MinariConverter,
};
use anyhow::Result;
use border_core::generic_replay_buffer::BatchBase;
use candle_core::{DType, Device, Tensor};
use ndarray::{s, ArrayBase, ArrayD, Axis, IxDyn, Slice};
use numpy::array;
use pyo3::{PyAny, PyObject};

/// State of the Kitchen environment represented by [`candle_core::Tensor`].
///
/// [`candle_core::Tensor`]: candle_core::Tensor
#[derive(Clone, Debug)]
pub struct KitchenState {
    pub kettle: Tensor,
}

impl KitchenState {
    pub fn new(capacity: usize) -> Self {
        Self {
            kettle: Tensor::zeros((capacity, 7), DType::F32, &Device::Cpu).unwrap(),
        }
    }

    pub fn push(&mut self, ix: usize, data: Self) {
        self.kettle.slice_set(&data.kettle, 0, ix).unwrap();
    }

    pub fn sample(&self, ixs: &Vec<usize>) -> Self {
        let ixs = Tensor::from_vec(
            ixs.iter().map(|&ix| ix as u32).collect::<Vec<u32>>(),
            (ixs.len(),),
            &Device::Cpu,
        )
        .unwrap();
        Self {
            kettle: self.kettle.index_select(&ixs, 0).unwrap(),
        }
    }
}

/// Observation of the Kitchen environment represented by [`candle_core::Tensor`].
///
/// [`candle_core::Tensor`]: candle_core::Tensor
#[derive(Clone, Debug)]
pub struct KitchenObs {
    pub achieved_goal: KitchenState,
    pub desired_goal: KitchenState,
}

impl border_core::Obs for KitchenObs {
    fn dummy(n: usize) -> Self {
        Self {
            achieved_goal: KitchenState::new(n),
            desired_goal: KitchenState::new(n),
        }
    }

    fn len(&self) -> usize {
        self.achieved_goal.kettle.dims()[0]
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
    pub action: Tensor,
}

impl border_core::Act for KitchenAct {}

/// Batch of action.
#[derive(Debug)]
pub struct KitchenActBatch {
    pub action: Tensor,
}

impl KitchenActBatch {
    /// Returns an action at the specified index in the batch.
    pub fn get(&self, ix: usize) -> KitchenAct {
        KitchenAct {
            action: self
                .action
                .index_select(&(ix as u32).try_into().unwrap(), 0)
                .unwrap()
                .copy()
                .unwrap()
        }
    }
}

impl BatchBase for KitchenActBatch {
    fn new(capacity: usize) -> Self {
        Self {
            action: Tensor::zeros((capacity, 9), DType::F32, &Device::Cpu).unwrap(),
        }
    }

    fn push(&mut self, ix: usize, data: Self) {
        self.action.slice_set(&data.action, 0, ix).unwrap();
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let action = {
            let ixs = Tensor::from_vec(
                ixs.iter().map(|&ix| ix as u32).collect::<Vec<u32>>(),
                (ixs.len(),),
                &Device::Cpu,
            ).unwrap();
            self.action.index_select(&ixs, 0).unwrap().copy().unwrap()
        };

        Self { action }
    }
}

impl From<KitchenAct> for KitchenActBatch {
    fn from(act: KitchenAct) -> Self {
        Self { action: act.action }
    }
}

// Converter.
pub struct KitchenConverter {}

impl MinariConverter for KitchenConverter {
    type Obs = KitchenObs;
    type Act = KitchenAct;
    type ObsBatch = KitchenObsBatch;
    type ActBatch = KitchenActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let achieved_goal = obj.get_item("achieved_goal")?;
        let desired_goal = obj.get_item("desired_goal")?;
        Ok(KitchenObs {
            achieved_goal: KitchenState {
                kettle: arrayd_to_tensor(pyobj_to_arrayd::<f64, f32>(
                    achieved_goal.get_item("kettle")?.extract()?,
                ))?,
            },
            desired_goal: KitchenState {
                kettle: arrayd_to_tensor(pyobj_to_arrayd::<f64, f32>(
                    desired_goal.get_item("kettle")?.extract()?,
                ))?,
            },
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(tensor_to_arrayd(act.action)?))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let achieved_goal = obj.get_item("achieved_goal")?;
        let desired_goal = obj.get_item("desired_goal")?;
        Ok(KitchenObsBatch {
            achieved_goal: KitchenState {
                kettle: pyobj_to_tensor1(achieved_goal, "kettle")?,
            },
            desired_goal: KitchenState {
                kettle: pyobj_to_tensor1(desired_goal, "kettle")?,
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
                kettle: pyobj_to_tensor2(achieved_goal, "kettle")?,
            },
            desired_goal: KitchenState {
                kettle: pyobj_to_tensor2(desired_goal, "kettle")?,
            },
        })
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(KitchenActBatch {
            action: {
                let arr = pyobj_to_arrayd::<f64, f32>(obj.into());
                arrayd_to_tensor(arr)?
            },
        })
    }
}

/// Converts PyObject to [`candle_core::Tensor`] and drop the last row.
fn pyobj_to_tensor1(obj: &PyAny, name: &str) -> Result<Tensor> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);

    // Drop the last row
    let arr = arr.slice_axis(Axis(0), Slice::from(..-1)).to_owned();

    // Convert to Tensor
    Ok(arrayd_to_tensor(arr)?)
}

/// Converts PyObject to Tensor and drop the first row.
fn pyobj_to_tensor2(obj: &PyAny, name: &str) -> Result<Tensor> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);

    // Drop the first row
    let arr = arr.slice_axis(Axis(0), Slice::from(1..)).to_owned();

    // Convert to Tensor
    Ok(arrayd_to_tensor(arr)?)
}

/// Converts ArrayD to tensor.
fn arrayd_to_tensor(arr: ArrayD<f32>) -> Result<Tensor> {
    let tensor =
        Tensor::try_from(arr.as_slice().expect("Slice of ndarray"))?.reshape(arr.shape())?;
    Ok(tensor)
}

/// Converts tensor to ArrayD.
fn tensor_to_arrayd(tensor: Tensor) -> Result<ArrayD<f32> > {
    let shape = tensor.dims().iter().map(|&x| x as usize).collect::<Vec<usize>>();
    let arr = ArrayBase::from_vec(tensor.to_vec1()?).into_shape(shape)?;
    Ok(arr)
}
