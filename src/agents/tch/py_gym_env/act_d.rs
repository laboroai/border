use std::convert::TryFrom;
use log::trace;
use pyo3::{PyObject, IntoPy};
use tch::Tensor;
use crate::core::Act;
use crate::agents::tch::TchBuffer;

#[derive(Clone, Debug)]
pub struct TchPyGymEnvDiscreteAct (Vec<i32>);

impl Act for TchPyGymEnvDiscreteAct {}

impl Into<PyObject> for TchPyGymEnvDiscreteAct {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            self.0.into_py(py)
        })
    }
}

impl From<Tensor> for TchPyGymEnvDiscreteAct {
    /// Assumes `t` is a scalar or 1-dimensional vector,
    /// containing discrete actions. The number of elements is
    /// equal to the number of environments in the vectorized environment.
    fn from(t: Tensor) -> Self {
        trace!("Tensor from TchPyGymEnvDiscreteAct: {:?}", t);
        Self(t.into())
    }
}

pub struct TchPyGymEnvDiscreteActBuffer {
    act: Tensor
}

// // Borrowed from tch-rs. The original code didn't work with ndarray 0.14.
// fn try_from(value: ndarray::Array<$type, D>) -> Result<Self, Self::Error> {
//     // TODO: Replace this with `?` once it works with `std::option::ErrorNone`
//     let slice = match value.as_slice() {
//         None => return Err(TchError::Convert("cannot convert to slice".to_string())),
//         Some(v) => v,
//     };
//     let tn = Self::f_of_slice(slice)?;
//     let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
//     Ok(tn.f_reshape(&shape)?)
// }

impl TchBuffer for TchPyGymEnvDiscreteActBuffer {
    type Item = TchPyGymEnvDiscreteAct;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        Self {
            act: Tensor::zeros(&[capacity as _, n_procs as _], tch::kind::INT64_CPU),
        }
    }

    fn push(&mut self, index: i64, item: &TchPyGymEnvDiscreteAct) {
        let act = Tensor::try_from(item.0.clone()).unwrap();
        trace!("TchPyGymDiscreteActBuffer.push(): {:?}", act);
        self.act.get(index).copy_(&act);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        let batch = self.act.index_select(0, &batch_indexes);
        batch.flatten(0, 1)
    }
}
