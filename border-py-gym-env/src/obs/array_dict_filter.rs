use crate::util::{Array, ArrayType};
use crate::GymObsFilter;
use border_core::{
    record::{Record, RecordValue},
    Obs,
};
use pyo3::PyObject;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Configuration of [`ArrayDictObsFilter`].
pub struct ArrayDictObsFilterConfig {
    pub key_and_types: Vec<(String, ArrayType)>,

    /// If `None`, no values will be recorded.
    pub record_keys: Option<Vec<String>>,
}

impl Default for ArrayDictObsFilterConfig {
    fn default() -> Self {
        Self {
            key_and_types: vec![],
            record_keys: None,
        }
    }
}

/// An observation filter for dict of arrays.
pub struct ArrayDictObsFilter<O>
where
    O: Obs + From<Vec<(String, Array)>>,
{
    config: ArrayDictObsFilterConfig,

    /// Marker.
    phantom: PhantomData<O>,
}

impl<O> Default for ArrayDictObsFilter<O>
where
    O: Obs + From<Vec<(String, Array)>>,
{
    fn default() -> Self {
        Self {
            config: ArrayDictObsFilterConfig::default(),
            phantom: PhantomData,
        }
    }
}

impl<O> GymObsFilter<O> for ArrayDictObsFilter<O>
where
    O: Obs + From<Vec<(String, Array)>>,
{
    type Config = ArrayDictObsFilterConfig;

    fn build(config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            config: config.clone(),
            phantom: PhantomData,
        })
    }

    /// Convert `PyObject` to an obervation.
    ///
    /// This method assumes that `obj` is a Python dict.
    ///
    /// [Record] in the returned value has `obs`, which is a flattened array of
    /// observation, for either of single and vectorized environments.
    fn filt(&mut self, obs: PyObject) -> (O, Record)
    where
        O: From<Vec<(String, Array)>>
    {
        let obs = pyo3::Python::with_gil(|py| {
            self.config
                .key_and_types
                .iter()
                .map(|(key, t)| {
                    let pyobj = obs.call_method1(py, "get", (key,)).unwrap();
                    (key.clone(), t.to_array(pyobj))
                })
                .collect::<Vec<_>>()
        });
        let record = match &self.config.record_keys {
            None => Record::empty(),
            Some(keys) => {
                let mut record = Record::empty();
                obs.iter().for_each(|(key, arr)| {
                    if keys.contains(key) {
                        let v = arr.to_flat_vec::<f32>();
                        record.insert(key, RecordValue::Array1(v))    
                    }
                });
                record    
            }
        };
        (obs.into(), record)
    }
}
