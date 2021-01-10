use std::marker::PhantomData;
use pyo3::{PyObject};
use ndarray::{ArrayD, Axis, IxDyn};
use numpy::PyArrayDyn;
use tch::{Tensor, TchError};
use crate::core::Obs;

fn any(is_done: &[f32]) -> bool {
    is_done.iter().fold(0, |x, v| x + *v as i32) > 0
}

pub trait Shape: Clone {
    fn shape() -> &'static [usize];
}

#[derive(Clone, Debug)]
pub struct TchPyGymEnvObs<S: Shape> {
    obs: ArrayD<f32>,
    phantom: PhantomData<S>
}

impl<S: Shape> Obs for TchPyGymEnvObs<S> {
    fn zero(n_procs: usize) -> Self {
        let shape = &mut S::shape().to_vec();
        shape.insert(0, n_procs as _);
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData
        }
    }

    fn merge(mut self, obs_reset: Self, is_done: &[f32]) -> Self {
        if any(is_done) {
            for (i, is_done_i) in is_done.iter().enumerate() {
                if *is_done_i != 0.0 as f32 {
                    self.obs.index_axis_mut(Axis(0), i)
                        .assign(&obs_reset.obs.index_axis(Axis(0), i));
                }
            }
        };
        self
    }
}

impl<S: Shape> From<PyObject> for TchPyGymEnvObs<S> {
    fn from(obs: PyObject) -> Self {
        pyo3::Python::with_gil(|py| {
            let obs: &PyArrayDyn<f64> = obs.extract(py).unwrap();
            let obs = obs.to_owned_array();
            let obs = obs.mapv(|elem| elem as f32);
            let obs = {
                if obs.shape().len() > S::shape().len() {
                    obs.insert_axis(Axis(0))
                }
                else {
                    obs
                }
            };
            Self {
                obs,
                phantom: PhantomData,
            }
        })
    }
}

// Borrowed from tch-rs. The original code didn't work with ndarray 0.14.
fn try_from(value: ArrayD<f32>) -> Result<Tensor, TchError> {
    // TODO: Replace this with `?` once it works with `std::option::ErrorNone`
    let slice = match value.as_slice() {
        None => return Err(TchError::Convert("cannot convert to slice".to_string())),
        Some(v) => v,
    };
    let tn = Tensor::f_of_slice(slice)?;
    let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
    Ok(tn.f_reshape(&shape)?)
}

impl<S: Shape> Into<Tensor> for TchPyGymEnvObs<S> {
    fn into(self) -> Tensor {
        try_from(self.obs).unwrap()
    }
}
