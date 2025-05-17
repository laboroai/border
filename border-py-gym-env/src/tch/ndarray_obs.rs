use super::TensorBatch;
use ndarray::ArrayD;
use std::convert::TryFrom;
use tch::Tensor;

#[derive(Clone, Debug)]
/// Observation.
pub struct NdarrayObs(pub ArrayD<f32>);

impl border_core::Obs for NdarrayObs {
    fn len(&self) -> usize {
        self.0.shape()[0]
    }
}

impl Into<Tensor> for NdarrayObs {
    fn into(self) -> Tensor {
        Tensor::try_from(&self.0).unwrap()
    }
}

impl From<NdarrayObs> for TensorBatch {
    fn from(o: NdarrayObs) -> Self {
        TensorBatch::from_tensor(o.into())
    }
}
