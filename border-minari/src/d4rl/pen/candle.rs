//! Observation, action types and corresponding converters for the Pen environment implemented with candle.
use crate::{
    util::ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    MinariConverter, MinariDataset,
};
use anyhow::Result;
use border_core::generic_replay_buffer::BatchBase;
use candle_core::{DType, Device, Tensor};
use ndarray::{ArrayBase, ArrayD, Axis, Slice};
use pyo3::{types::PyIterator, PyAny, PyObject, Python};
use std::convert::{TryFrom, TryInto};

mod obs {
    use super::Tensor;

    /// Observation of the Pen environment stored as [`Tensor`].
    ///
    /// To create of batch of observations, this struct can be converted into [`PenObsBatch`].
    ///
    /// [`Tensor`]: candle_core::Tensor
    #[derive(Clone, Debug)]
    pub struct PenObs {
        pub(super) obs: Tensor,
    }

    impl border_core::Obs for PenObs {
        fn len(&self) -> usize {
            self.obs.dims()[0]
        }
    }

    /// Converts [`PenObs`] to Tensor.
    impl Into<Tensor> for PenObs {
        fn into(self) -> Tensor {
            self.obs
        }
    }
}

mod obs_batch {
    use super::{BatchBase, DType, Device, PenObs, Tensor};

    /// Batch of observations.
    ///
    /// It can be converted from [`PenObs`].
    ///
    /// It can be converted into [`Tensor`].　This allows a batch of observations to be fed into a neural network.
    ///
    /// [`Tensor`]: candle_core::Tensor
    #[derive(Clone, Debug)]
    pub struct PenObsBatch {
        pub(super) capacity: usize,
        pub(super) obs: Option<Tensor>,
    }

    impl BatchBase for PenObsBatch {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                obs: None,
                // obs: ,
            }
        }

        fn push(&mut self, ix: usize, data: Self) {
            // Push samples when data is not empty
            if let Some(obs) = &data.obs {
                // Lazy creation of the internal buffer
                if self.obs.is_none() {
                    let dim = obs.dims()[1];
                    self.obs = Some(
                        Tensor::zeros((self.capacity, dim), DType::F32, &Device::Cpu).unwrap(),
                    );
                }

                // Push samples to the internal buffer
                self.obs.as_mut().unwrap().slice_set(&obs, 0, ix).unwrap();
            }
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let capacity = ixs.len();
            let ixs = Tensor::from_vec(
                ixs.iter().map(|&ix| ix as u32).collect::<Vec<u32>>(),
                (ixs.len(),),
                &Device::Cpu,
            )
            .unwrap();
            Self {
                capacity,
                obs: Some(self.obs.as_ref().unwrap().index_select(&ixs, 0).unwrap()),
            }
        }
    }

    impl From<PenObs> for PenObsBatch {
        fn from(obs: PenObs) -> Self {
            // Size of obs = [batch_size, dim_of_obs_vec]
            assert_eq!(obs.obs.dims().len(), 24);

            Self {
                capacity: obs.obs.dims()[0],
                obs: Some(obs.obs),
            }
        }
    }

    impl Into<Tensor> for PenObsBatch {
        fn into(self) -> Tensor {
            self.obs.unwrap()
        }
    }
}

mod act {
    use super::Tensor;

    /// Action of the Pen environment stored as [`Tensor`].
    ///
    /// It can be converted from a [`Tensor`] and can be converted into a [`PyObject`].
    /// It allows the action to inferred from the neural network and be passed to the Python interpreter.
    ///
    /// To create a batch of actions, this struct can be converted into [`PenActBatch`].
    ///
    /// [`Tensor`]: candle_core::Tensor
    #[derive(Clone, Debug)]
    pub struct PenAct {
        pub(super) action: Tensor,
    }

    impl border_core::Act for PenAct {}

    impl From<Tensor> for PenAct {
        fn from(action: Tensor) -> Self {
            Self { action }
        }
    }

    impl Into<Tensor> for PenAct {
        fn into(self) -> Tensor {
            self.action
        }
    }
}

mod act_batch {
    use super::{BatchBase, DType, Device, PenAct, Tensor, TryInto};

    /// Batch of actions.
    ///
    /// It can be converted into [`Tensor`] for handling with neural networks.
    ///
    /// [`Tensor`]: candle_core::Tensor
    #[derive(Clone, Debug)]
    pub struct PenActBatch {
        pub(super) action: Tensor,
    }

    impl PenActBatch {
        /// Returns an action at the specified index in the batch.
        pub fn get(&self, ix: usize) -> PenAct {
            PenAct {
                action: self
                    .action
                    .index_select(&(ix as u32).try_into().unwrap(), 0)
                    .unwrap()
                    .copy()
                    .unwrap(),
            }
        }
    }

    impl BatchBase for PenActBatch {
        fn new(capacity: usize) -> Self {
            Self {
                // Dimension of action vector should be 2
                action: Tensor::zeros((capacity, 24), DType::F32, &Device::Cpu).unwrap(),
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

    impl From<PenAct> for PenActBatch {
        fn from(act: PenAct) -> Self {
            Self { action: act.action }
        }
    }

    impl Into<Tensor> for PenActBatch {
        fn into(self) -> Tensor {
            self.action
        }
    }
}

pub use act::PenAct;
pub use act_batch::PenActBatch;
pub use obs::PenObs;
pub use obs_batch::PenObsBatch;

/// Configuration of [`PenConverter`].
pub struct PenConverterConfig {}

impl Default for PenConverterConfig {
    fn default() -> Self {
        Self {}
    }
}

/// Converter for the Pen environment implemented with candle.
///
/// This struct normalizes observations based on the statistics
/// of the observations in the dataset.
pub struct PenConverter {
    mean: Tensor, // for normalizing observation
    std: Tensor,  // for normalizing observation
}

impl PenConverter {
    /// Creates a new Pen converter.
    ///
    /// `dataset` is used to calculate the mean and standard deviation of the observations.
    pub fn new(_config: PenConverterConfig, dataset: &MinariDataset) -> Result<Self> {
        let (mean, std) = Python::with_gil(|py| -> Result<(Tensor, Tensor)> {
            // Iterate all episodes
            let episodes = dataset
                .dataset
                .call_method1(py, "iterate_episodes", (None::<i32>,))?;
            let mut all_obs = Tensor::zeros(&[0, 45], DType::F32, &Device::Cpu)?;

            // Collect all observations for calculating mean and std
            for ep in PyIterator::from_object(py, &episodes)? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;
                let obj = ep.getattr("observations")?;

                let obs_batch = pyobj_to_tensor1(obj.into())?;
                all_obs = Tensor::cat(&[all_obs, obs_batch], 0)?;
            }

            // Calculate mean and std
            let mean = all_obs.mean(0)?.unsqueeze(0)?;
            let std = all_obs.var(0)?.sqrt()?.unsqueeze(0)?;
            debug_assert_eq!(mean.dims(), &[1, 45]);

            Ok((mean, std))
        })?;

        Ok(Self { mean, std })
    }

    fn normalize_observation(&self, obs: &Tensor) -> Result<Tensor> {
        Ok(obs.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}

impl MinariConverter for PenConverter {
    type Obs = PenObs;
    type Act = PenAct;
    type ObsBatch = PenObsBatch;
    type ActBatch = PenActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let obs = arrayd_to_tensor(pyobj_to_arrayd::<f64, f32>(obj.into()), Some(&[1, 45]))?;
        Ok(PenObs {
            obs: self.normalize_observation(&obs)?,
        })
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        Ok(arrayd_to_pyobj(tensor_to_arrayd(act.action)?))
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let obs = pyobj_to_tensor1(obj)?;
        let obs = self.normalize_observation(&obs)?;

        // Check tensor size: expects [batch_size, obs_vec_dim]
        let batch_size = obs.dims()[0];
        debug_assert_eq!(obs.dims(), &[batch_size, 45]);

        Ok(PenObsBatch {
            capacity: batch_size,
            obs: Some(obs),
        })
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let obs = pyobj_to_tensor2(obj)?;
        let obs = self.normalize_observation(&obs)?;

        // Check tensor size: expects [batch_size, obs_vec_dim]
        let batch_size = obs.dims()[0];
        debug_assert_eq!(obs.dims(), &[batch_size, 45]);

        Ok(PenObsBatch {
            capacity: batch_size,
            obs: Some(obs),
        })
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(PenActBatch {
            action: {
                let arr = pyobj_to_arrayd::<f32, f32>(obj.into());
                arrayd_to_tensor(arr, None)?
            },
        })
    }

    fn env_params(&self) -> Vec<(&str, Option<&str>)> {
        // not override the original parameters in Minari
        vec![]

        // When want to override the parameters, comment in the following code:
        // vec![
        //     ("continuing_task", Some("True")),
        //     ("reset_target", Some("True")),
        // ]
    }
}

/// Converts PyObject to [`candle_core::Tensor`] and drop the last row.
fn pyobj_to_tensor1(obj: &PyAny) -> Result<Tensor> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.into());

    // Drop the last row
    let arr = arr.slice_axis(Axis(0), Slice::from(..-1)).to_owned();

    // Convert to Tensor
    Ok(arrayd_to_tensor(arr, None)?)
}

/// Converts PyObject to Tensor and drop the first row.
fn pyobj_to_tensor2(obj: &PyAny) -> Result<Tensor> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.into());

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
