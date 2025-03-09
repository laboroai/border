//! Observation, action types and corresponding converters for the Point Maze environment implemented with candle.
use crate::{
    util::{
        candle::{NdarrayAct, NdarrayObs, TensorBatch},
        ndarray::{arrayd_to_pyobj, pyobj_to_arrayd},
    },
    MinariConverter, MinariDataset,
};
use anyhow::Result;
use candle_core::Tensor;
use ndarray::{concatenate, ArrayD, Axis, IxDyn, Slice};
use pyo3::{types::PyIterator, PyAny, PyObject, Python, ToPyObject};
use std::convert::TryFrom;

pub type PointMazeAct = NdarrayAct;
pub type PointMazeObs = NdarrayObs;
pub type PointMazeActBatch = TensorBatch;
pub type PointMazeObsBatch = TensorBatch;

/// Configuration of [`PointMazeConverter`].
pub struct PointMazeConverterConfig {
    /// If `true`, the observation vectors will include the x and y positions in the last two dimensions.
    /// Default is `false`.
    pub include_goal: bool,
}

impl Default for PointMazeConverterConfig {
    fn default() -> Self {
        Self {
            include_goal: false,
        }
    }
}

impl PointMazeConverterConfig {
    pub fn include_goal(self, value: bool) -> Self {
        let mut config = self;
        config.include_goal = value;
        config
    }
}

/// Converter for the Point Maze environment implemented with candle.
///
/// This struct normalizes observations based on the statistics
/// of the observations in the dataset.
pub struct PointMazeConverter {
    include_goal: bool,
    mean: ArrayD<f32>, // for normalizing observation
    std: ArrayD<f32>,  // for normalizing observation
}

impl PointMazeConverter {
    /// Creates a new Point Maze converter.
    ///
    /// `dataset` is used to calculate the mean and standard deviation of the observations.
    pub fn new(config: PointMazeConverterConfig, dataset: &MinariDataset) -> Result<Self> {
        let (mean, std) = Python::with_gil(|py| -> Result<(ArrayD<f32>, ArrayD<f32>)> {
            // Iterate all episodes
            let episodes = dataset
                .dataset
                .call_method1(py, "iterate_episodes", (None::<i32>,))?;
            let mut all_obs = ArrayD::<f32>::zeros(IxDyn(&[0, 4]));

            // Collect all observations for calculating mean and std
            for ep in PyIterator::from_object(py, &episodes)? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;
                let obj = ep.getattr("observations")?;

                let obs_batch = pyobj_to_ndarray1(obj, "observation")?;
                all_obs = concatenate![Axis(0), all_obs, obs_batch];
            }

            // Calculate mean and std
            let mean = all_obs.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
            let std = all_obs.std_axis(Axis(0), 1.0).insert_axis(Axis(0));
            debug_assert_eq!(mean.shape(), &[1, 4]);

            Ok((mean, std))
        })?;

        Ok(Self {
            include_goal: config.include_goal,
            mean,
            std,
        })
    }

    fn normalize_observation(&self, obs: &NdarrayObs) -> Result<NdarrayObs> {
        let normalized_obs = (&obs.0 - &self.mean) / &self.std;
        Ok(normalized_obs.into())
    }
}

impl MinariConverter for PointMazeConverter {
    type Obs = PointMazeObs;
    type Act = PointMazeAct;
    type ObsBatch = PointMazeObsBatch;
    type ActBatch = PointMazeActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        match self.include_goal {
            false => {
                let obs = obj.get_item("observation")?.extract()?;
                let obs = NdarrayObs(pyobj_to_arrayd::<f64, f32>(obs));
                Ok(self.normalize_observation(&obs)?)
            }
            true => {
                todo!();
                // let obs = obj.get_item("observation")?.extract()?;
                // let obs = arrayd_to_tensor(pyobj_to_arrayd::<f64, f32>(obs), Some(&[1, 4]))?;
                // let obs = self.normalize_observation(&obs)?;
                // let goal = obj.get_item("desired_goal")?.extract()?;
                // let goal = arrayd_to_tensor(pyobj_to_arrayd::<f64, f32>(goal), Some(&[1, 2]))?;
                // Ok(PointMazeObs {
                //     obs: Tensor::cat(&[obs, goal], candle_core::D::Minus1)?,
                // })
            }
        }
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        let act = match act {
            NdarrayAct::Continuous(act) => act,
            NdarrayAct::Discrete(_) => {
                panic!("PointMazeConverter does not support discrete action.");
            }
        };

        // println!("{:?}", act);
        let act2 = arrayd_to_pyobj(act);
        // println!("{:?}", act2);

        Ok(act2)
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        match self.include_goal {
            false => {
                let obs = pyobj_to_ndarray1(obj, "observation")?;
                let obs = self.normalize_observation(&NdarrayObs(obs))?;

                // Check tensor size: expects [batch_size, obs_vec_dim]
                let batch_size = obs.0.shape()[0];
                debug_assert_eq!(obs.0.shape(), &[batch_size, 4]);

                Ok(PointMazeObsBatch::from(arrayd_to_tensor(obs.0, None)?))
            }
            true => {
                todo!();
                // let obs = pyobj_to_tensor1(obj, "observation")?;
                // let goal = pyobj_to_tensor1(obj, "desired_goal")?;

                // // Drop the last dim
                // let obs = obs.squeeze(candle_core::D::Minus1)?;
                // let goal = goal.squeeze(candle_core::D::Minus1)?;

                // // Normalize obs (keep goal unchanged)
                // let obs = self.normalize_observation(&obs)?;

                // // Check tensor size: expects [batch_size, obs_vec_dim]
                // let batch_size = obs.dims()[0];
                // debug_assert_eq!(obs.dims(), &[batch_size, 4]);
                // debug_assert_eq!(goal.dims(), &[batch_size, 2]);

                // // Concat obs and goal
                // let obs = Tensor::cat(&[obs, goal], candle_core::D::Minus1)?;

                // Ok(PointMazeObsBatch {
                //     capacity: batch_size,
                //     obs: Some(obs),
                // })
            }
        }
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        match self.include_goal {
            false => {
                let obs = pyobj_to_ndarray2(obj, "observation")?;
                let obs = self.normalize_observation(&NdarrayObs(obs))?;

                // Check tensor size: expects [batch_size, obs_vec_dim]
                let batch_size = obs.0.shape()[0];
                debug_assert_eq!(obs.0.shape(), &[batch_size, 4]);

                Ok(PointMazeObsBatch::from(arrayd_to_tensor(obs.0, None)?))
            }
            true => {
                todo!();
                // let obs = pyobj_to_tensor2(obj, "observation")?;
                // let goal = pyobj_to_tensor2(obj, "desired_goal")?;

                // // Drop the last dim
                // let obs = obs.squeeze(candle_core::D::Minus1)?;
                // let goal = goal.squeeze(candle_core::D::Minus1)?;

                // // Normalize
                // let obs = self.normalize_observation(&obs)?;

                // // Check tensor size: expects [batch_size, obs_vec_dim]
                // let batch_size = obs.dims()[0];
                // debug_assert_eq!(obs.dims(), &[batch_size, 4]);
                // debug_assert_eq!(goal.dims(), &[batch_size, 2]);

                // // Concat obs and goal
                // let obs = Tensor::cat(&[obs, goal], candle_core::D::Minus1)?;

                // Ok(PointMazeObsBatch {
                //     capacity: batch_size,
                //     obs: Some(obs),
                // })
            }
        }
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(PointMazeActBatch::from({
            let arr = pyobj_to_arrayd::<f32, f32>(obj.into());
            arrayd_to_tensor(arr, None)?
        }))
    }

    fn env_params(&self, py: Python<'_>) -> Vec<(&str, PyObject)> {
        // not override the original parameters in Minari
        // https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/cc54b71147650b310f5a84c642dd6dc127f333a1/scripts/pointmaze/create_pointmaze_dataset.py#L157-L159
        vec![("max_episode_steps", 300.to_object(py))]

        // When want to override the parameters, comment in the following code:
        // vec![
        //     ("continuing_task", Some("True")),
        //     ("reset_target", Some("True")),
        // ]
    }
}

/// Converts PyObject to `NdArray` and drop the last row.
fn pyobj_to_ndarray1(obj: &PyAny, name: &str) -> Result<ArrayD<f32>> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);

    // Drop the last row
    Ok(arr.slice_axis(Axis(0), Slice::from(..-1)).to_owned())
}

/// Converts PyObject to `NdArray` and drop the first row.
fn pyobj_to_ndarray2(obj: &PyAny, name: &str) -> Result<ArrayD<f32>> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.get_item(name)?.extract()?);

    // Drop the last row
    Ok(arr.slice_axis(Axis(0), Slice::from(1..)).to_owned())
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
