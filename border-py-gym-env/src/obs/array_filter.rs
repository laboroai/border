use crate::{GymObsFilter, util::pyobj_to_arrayd};
use border_core::{
    record::{Record, RecordValue},
    Obs,
};
use ndarray::ArrayD;
use num_traits::cast::AsPrimitive;
use numpy::Element;
use pyo3::PyObject;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [`ArrayObsFilter`].
#[derive(Clone)]
pub struct ArrayObsFilterConfig {}

impl Default for ArrayObsFilterConfig {
    fn default() -> Self {
        Self {}
    }
}

/// An observation filter that convertes PyObject of an numpy array.
///
/// Type parameter `O` must implements [`From`]`<ArrayD>` and [`border_core::Obs`].
pub struct ArrayObsFilter<T1, T2, O> {
    /// Marker.
    pub phantom: PhantomData<(T1, T2, O)>,
}

impl<T1, T2, O> Default for ArrayObsFilter<T1, T2, O> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T1, T2, O> GymObsFilter<O> for ArrayObsFilter<T1, T2, O>
where
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + Debug + num_traits::Zero + AsPrimitive<f32>,
    O: Obs + From<ArrayD<T2>>,
{
    type Config = ArrayObsFilterConfig;

    fn build(_config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            phantom: PhantomData,
        })
    }

    /// Convert `PyObject` to an obervation, which can be converted from [`ArrayD`].
    ///
    /// [Record] in the returned value has `obs`, which is a flattened array of
    /// observation, for either of single and vectorized environments.
    fn filt(&mut self, obs: PyObject) -> (O, Record) {
        let obs = pyo3::Python::with_gil(|py| {
            if obs.as_ref(py).get_type().name().unwrap() == "NoneType" {
                panic!();
            } else {
                pyobj_to_arrayd::<T1, T2>(obs)
            }
        });
        let record = {
            // let vec = Vec::<f32>::from_iter(obs.obs.iter().map(|x| x.as_()));
            let vec = obs.iter().map(|x| x.as_()).collect();
            Record::from_slice(&[("obs", RecordValue::Array1(vec))])
        };
        (obs.into(), record)
    }
}
