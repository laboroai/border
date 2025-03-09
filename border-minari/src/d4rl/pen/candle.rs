//! Observation, action types and corresponding converters for the Pen environment implemented with candle.
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
use pyo3::{types::PyIterator, PyAny, PyObject, Python};
use std::convert::TryFrom;

pub type PenAct = NdarrayAct;
pub type PenObs = NdarrayObs;
pub type PenActBatch = TensorBatch;
pub type PenObsBatch = TensorBatch;

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
    mean: ArrayD<f32>, // for normalizing observation
    std: ArrayD<f32>,  // for normalizing observation
}

impl PenConverter {
    /// Creates a new Pen converter.
    ///
    /// `dataset` is used to calculate the mean and standard deviation of the observations.
    pub fn new(_config: PenConverterConfig, dataset: &MinariDataset) -> Result<Self> {
        let (mean, std) = Python::with_gil(|py| -> Result<(ArrayD<f32>, ArrayD<f32>)> {
            // Iterate all episodes
            let episodes = dataset
                .dataset
                .call_method1(py, "iterate_episodes", (None::<i32>,))?;
            let mut all_obs = ArrayD::<f32>::zeros(IxDyn(&[0, 45]));

            // Collect all observations for calculating mean and std
            for ep in PyIterator::from_object(py, &episodes)? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;
                let obj = ep.getattr("observations")?;

                let obs_batch = pyobj_to_ndarray1(obj)?;
                all_obs = concatenate![Axis(0), all_obs, obs_batch];
            }

            // Calculate mean and std
            let mean = all_obs.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
            let std = all_obs.std_axis(Axis(0), 1.0).insert_axis(Axis(0));
            debug_assert_eq!(mean.shape(), &[1, 45]);

            Ok((mean, std))
        })?;

        Ok(Self { mean, std })
    }

    fn normalize_observation(&self, obs: &NdarrayObs) -> Result<NdarrayObs> {
        let normalized_obs = (&obs.0 - &self.mean) / &self.std;
        Ok(normalized_obs.into())
    }
}

impl MinariConverter for PenConverter {
    type Obs = PenObs;
    type Act = PenAct;
    type ObsBatch = PenObsBatch;
    type ActBatch = PenActBatch;

    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs> {
        let obs = obj.extract()?;
        let obs = NdarrayObs(pyobj_to_arrayd::<f64, f32>(obs));
        Ok(self.normalize_observation(&obs)?)
    }

    fn convert_action(&self, act: Self::Act) -> Result<PyObject> {
        let act = match act {
            NdarrayAct::Continuous(act) => act,
            NdarrayAct::Discrete(_) => {
                panic!("PenConverter does not support discrete action.");
            }
        };

        // println!("{:?}", act);
        let act2 = arrayd_to_pyobj(act);
        // println!("{:?}", act2);

        Ok(act2)
    }

    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let obs = pyobj_to_ndarray1(obj)?;
        let obs = self.normalize_observation(&NdarrayObs(obs))?;

        // Check tensor size: expects [batch_size, obs_vec_dim]
        let batch_size = obs.0.shape()[0];
        debug_assert_eq!(obs.0.shape(), &[batch_size, 45]);

        Ok(PenObsBatch::from(arrayd_to_tensor(obs.0, None)?))
    }

    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch> {
        let obs = pyobj_to_ndarray2(obj)?;
        let obs = self.normalize_observation(&NdarrayObs(obs))?;

        // Check tensor size: expects [batch_size, obs_vec_dim]
        let batch_size = obs.0.shape()[0];
        debug_assert_eq!(obs.0.shape(), &[batch_size, 4]);

        Ok(PenObsBatch::from(arrayd_to_tensor(obs.0, None)?))
    }

    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch> {
        Ok(PenActBatch::from({
            let arr = pyobj_to_arrayd::<f32, f32>(obj.into());
            arrayd_to_tensor(arr, None)?
        }))
    }

    fn env_params(&self, _py: Python<'_>) -> Vec<(&str, PyObject)> {
        // not override the original parameters in Minari
        vec![]

        // When want to override the parameters, comment in the following code:
        // vec![
        //     ("continuing_task", Some("True")),
        //     ("reset_target", Some("True")),
        // ]
    }
}

/// Converts PyObject to `NdArray` and drop the last row.
fn pyobj_to_ndarray1(obj: &PyAny) -> Result<ArrayD<f32>> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.into());

    // Drop the last row
    Ok(arr.slice_axis(Axis(0), Slice::from(..-1)).to_owned())
}

/// Converts PyObject to `NdArray` and drop the first row.
fn pyobj_to_ndarray2(obj: &PyAny) -> Result<ArrayD<f32>> {
    // From python object to ndarray
    let arr = pyobj_to_arrayd::<f64, f32>(obj.into());

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
