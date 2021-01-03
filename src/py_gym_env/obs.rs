use std::fmt::Debug;
use ndarray::{ArrayD, IxDyn};
use crate::core::Obs;

#[derive(Clone, Debug)]
pub struct PyNDArrayObs (pub ArrayD<f32>);

impl Obs for PyNDArrayObs {
    fn new() -> Self {
        PyNDArrayObs(ArrayD::<f32>::zeros(IxDyn(&[1])))
    }
}
