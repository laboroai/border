use super::{arrayd_to_tensor, TensorBatch};
use candle_core::Tensor;
use ndarray::ArrayD;

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
        arrayd_to_tensor::<_, f32>(self.0, false).unwrap()
    }
}

impl From<NdarrayObs> for TensorBatch {
    fn from(o: NdarrayObs) -> Self {
        TensorBatch::from_tensor(o.into())
    }
}
