use ndarray::ArrayD;

#[derive(Clone, Debug)]
/// Observation.
pub struct NdarrayObs(pub ArrayD<f32>);

impl border_core::Obs for NdarrayObs {
    fn len(&self) -> usize {
        self.0.shape()[0]
    }
}
