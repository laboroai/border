use tch::Tensor;
use crate::core::Obs;

pub trait TchObsAdapter<T: Obs> {
    /// Converts [crate::core::Env::Obs] to tch tensor.
    fn convert(&self, obs: &T) -> Tensor;

    /// Return the shape of tensors of observation.
    fn shape(&self) -> &[i64];
}

pub trait TchActAdapter<T> {
    /// Converts tch tensor to [crate::core::Env::Act].
    fn convert(&self, act: &Tensor) -> T;

    /// Convert back [crate::core::Env::Act] to tensor.
    fn back(&self, act: &T) -> Tensor;

    /// Return the shape of tensors of action.
    fn shape(&self) -> &[i64];
}
