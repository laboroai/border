use super::{arrayd_to_tensor, TensorBatch};
use candle_core::Tensor;
use ndarray::ArrayD;

#[derive(Clone, Debug)]
/// Observation of dict of [`ArrayD`].
///
/// Each array represents a vector and its type is `f32`.
pub struct NdarrayDictObs(pub Vec<(String, ArrayD<f32>)>);

impl border_core::Obs for NdarrayDictObs {
    fn len(&self) -> usize {
        match self.0.get(0) {
            Some(elem) => elem.1.shape()[0],
            None => 0,
        }
    }
}

impl Into<Tensor> for NdarrayDictObs {
    /// Converts [`NdarrayDictObs`] to a [`Tensor`].
    ///
    /// All elements of the dict will be concatenated.
    fn into(self) -> Tensor {
        let tensors: Vec<_> = self
            .0
            .into_iter()
            .map(|elem| arrayd_to_tensor::<_, f32>(elem.1, false).unwrap())
            .collect();
        Tensor::cat(&tensors, 1).unwrap()
    }
}

impl From<NdarrayDictObs> for TensorBatch {
    /// Converts [`NdarrayDictObs`] to a [`Tensor`].
    ///
    /// All elements of the dict will be concatenated.
    fn from(obs: NdarrayDictObs) -> TensorBatch {
        TensorBatch::from_tensor(obs.into())
    }
}
