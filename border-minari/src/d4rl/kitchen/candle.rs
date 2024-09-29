//! Observation, actiontypes and corresponding converters for the Kitchen environment impmented with candle.
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
use ndarray::{ArrayBase, ArrayD, Axis, Slice};
use pyo3::{PyAny, PyObject};

/// Observation of the Kitchen environment stored as [`Tensor`].
///
/// It contains a 59-dimensional vector as explained
/// [here](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/#observation-space).
///
/// To create of batch of observations, this struct can be converted into [`KitchenObsBatch`].
///
/// [`Tensor`]: candle_core::Tensor
#[derive(Clone, Debug)]
pub struct KitchenObs {
    pub obs: Tensor,
}

impl border_core::Obs for KitchenObs {
    fn dummy(n: usize) -> Self {
        Self {
            obs: Tensor::zeros((n, 59), DType::F32, &Device::Cpu).unwrap(),
        }
    }

    fn len(&self) -> usize {
        self.obs.dims()[0]
    }
}

/// Converts KitchenObs to Tensor by concatenating the achieved goal and the desired goal.
impl Into<Tensor> for KitchenObs {
    fn into(self) -> Tensor {
        self.obs
    }
}

/// Batch of observations.
///
/// It can be converted from an observation, i.e., instance of [`KitchenObs`].
///
/// It can be converted into [`Tensor`].　This allows a batch of observations to be fed into a neural network.
///
/// [`Tensor`]: candle_core::Tensor
#[derive(Debug)]
pub struct KitchenObsBatch {
    pub obs: Tensor,
}

impl BatchBase for KitchenObsBatch {
    fn new(capacity: usize) -> Self {
        Self {
            obs: Tensor::zeros((capacity, 59), DType::F32, &Device::Cpu).unwrap(),
        }
    }

    fn push(&mut self, ix: usize, data: Self) {
        self.obs.slice_set(&data.obs, 0, ix).unwrap();
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let ixs = Tensor::from_vec(
            ixs.iter().map(|&ix| ix as u32).collect::<Vec<u32>>(),
            (ixs.len(),),
            &Device::Cpu,
        )
        .unwrap();
        Self {
            obs: self.obs.index_select(&ixs, 0).unwrap(),
        }
    }
}

impl From<KitchenObs> for KitchenObsBatch {
    fn from(obs: KitchenObs) -> Self {
        Self { obs: obs.obs }
    }
}

impl Into<Tensor> for KitchenObsBatch {
    fn into(self) -> Tensor {
        self.obs
    }
}

/// Action of the Kitchen environment stored as [`Tensor`].
///
/// It can be converted from a [`Tensor`] and can be converted into a [`PyObject`].
/// It allows the action to inferred from the neural network and be passed to the Python interpreter.
///
/// To create a batch of actions, this struct can be converted into [`KitchenActBatch`].
///
/// [`Tensor`]: candle_core::Tensor
#[derive(Clone, Debug)]
pub struct KitchenAct {
    pub action: Tensor,
}

impl border_core::Act for KitchenAct {}

impl From<Tensor> for KitchenAct {
    fn from(action: Tensor) -> Self {
        Self { action }
    }
}

/// Batch of actions.
///
/// It can be converted into [`Tensor`].　This allows a batch of observations to be fed into a neural network.
///
/// [`Tensor`]: candle_core::Tensor
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
                .unwrap(),
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
            )
            .unwrap();
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

impl Into<Tensor> for KitchenActBatch {
    fn into(self) -> Tensor {
        self.action
    }
}

/// Converter for the Kitchen environment implemented with candle.
pub struct KitchenConverter {}

impl MinariConverter for KitchenConverter {
    type Obs = KitchenObs;
    type Act = KitchenAct;
    type ObsBatch = KitchenObsBatch;
    type ActBatch = KitchenActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let obs = obj.get_item("observation")?.extract()?;
        Ok(KitchenObs {
            obs: arrayd_to_tensor(pyobj_to_arrayd::<f64, f32>(obs), Some(&[1, 59]))?,
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(tensor_to_arrayd(act.action)?))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(KitchenObsBatch {
            obs: pyobj_to_tensor1(obj, "observation")?,
        })
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        Ok(KitchenObsBatch {
            obs: pyobj_to_tensor2(obj, "observation")?,
        })
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(KitchenActBatch {
            action: {
                let arr = pyobj_to_arrayd::<f64, f32>(obj.into());
                arrayd_to_tensor(arr, None)?
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
    Ok(arrayd_to_tensor(arr, None)?)
}

/// Converts PyObject to Tensor and drop the first row.
fn pyobj_to_tensor2(obj: &PyAny, name: &str) -> Result<Tensor> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);

    // Drop the first row
    let arr = arr.slice_axis(Axis(0), Slice::from(1..)).to_owned();

    // Convert to Tensor
    Ok(arrayd_to_tensor(arr, None)?)
}

/// Converts ArrayD to tensor.
///
/// When `shape` is None, its shape will be the result of `arr.shape()`.
fn arrayd_to_tensor(arr: ArrayD<f32>, shape: Option<&[usize]>) -> Result<Tensor> {
    let shape = match shape {
        Some(shape) => shape,
        None => arr.shape(),
    };
    let tensor = Tensor::try_from(arr.as_slice().expect("Slice of ndarray"))?.reshape(shape)?;
    Ok(tensor)
}

/// Converts tensor to ArrayD.
fn tensor_to_arrayd(tensor: Tensor) -> Result<ArrayD<f32>> {
    let shape = tensor
        .dims()
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();
    let arr = ArrayBase::from_vec(tensor.flatten_all()?.to_vec1()?).into_shape(shape)?;
    Ok(arr)
}
