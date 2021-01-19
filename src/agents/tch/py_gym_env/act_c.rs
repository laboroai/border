use std::marker::PhantomData;
use std::fmt::Debug;
use log::trace;
use pyo3::{PyObject, IntoPy};
use ndarray::{Array1, ArrayD, IxDyn};
use numpy::PyArrayDyn;
use tch::Tensor;
use crate::core::Act;
use crate::agents::tch::{Shape, TchBuffer, util::try_from, util::concat_slices};

/// Filtering action before applied to the environment.
pub trait TchPyGymActFilter: Clone + Debug {
    fn filter(act: ArrayD<f32>) -> ArrayD<f32> {
        act
    }
}

#[derive(Clone, Debug)]
pub struct RawFilter {}

impl TchPyGymActFilter for RawFilter {}

/// Represents action.
/// Currently, it supports 1-dimensional vector only.
#[derive(Clone, Debug)]
pub struct TchPyGymEnvContinuousAct<S: Shape, F: TchPyGymActFilter> {
    act: ArrayD<f32>,
    phantom: PhantomData<(S, F)>
}

impl<S: Shape, F: TchPyGymActFilter> TchPyGymEnvContinuousAct<S, F> {
    pub fn new(v: ArrayD<f32>) -> Self {
        Self {
            act: v,
            phantom: PhantomData
        }
    }
}

impl<S: Shape, F: TchPyGymActFilter> Act for TchPyGymEnvContinuousAct<S, F> {}

/// TODO: check action representation in the vectorized environment.
impl<S: Shape, F: TchPyGymActFilter> Into<PyObject> for TchPyGymEnvContinuousAct<S, F> {
    fn into(self) -> PyObject {
        let act = (&self.act).clone();
        let act = {
            if S::squeeze_first_dim() {
                debug_assert!(self.act.shape()[0] == 1);
                act.remove_axis(ndarray::Axis(0))
            }
            else {
                act
            }
        };
        let act = F::filter(act);
        pyo3::Python::with_gil(|py| {
            let act = PyArrayDyn::<f32>::from_array(py, &act);
            act.into_py(py)
        })
    }
}

impl<S: Shape, F: TchPyGymActFilter> From<Tensor> for TchPyGymEnvContinuousAct<S, F> {
    /// The first dimension is the number of environments.
    fn from(t: Tensor) -> Self {
        trace!("TchPyGymEnvContinuousAct from Tensor: {:?}", t);
        let shape = t.size().iter().map(|x| *x as usize).collect::<Vec<_>>();
        let act: Vec<f32> = t.into();
        Self {
            act: Array1::<f32>::from(act).into_shape(IxDyn(&shape)).unwrap(),
            phantom: PhantomData,
        }
    }
}

pub struct TchPyGymEnvContinuousActBuffer<S: Shape, F: TchPyGymActFilter> {
    act: Tensor,
    phantom: PhantomData<(S, F)>
}

impl<S: Shape, F: TchPyGymActFilter> TchBuffer for TchPyGymEnvContinuousActBuffer<S, F> {
    type Item = TchPyGymEnvContinuousAct<S, F>;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        let capacity = capacity as _;
        let n_procs = n_procs as _;
        let shape = concat_slices(&[capacity, n_procs],
            S::shape().iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice());
        Self {
            act: Tensor::zeros(&shape, tch::kind::FLOAT_CPU),
            phantom: PhantomData,
        }
    }

    fn push(&mut self, index: i64, item: &Self::Item) {
        let act = try_from(item.act.clone()).unwrap();
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
