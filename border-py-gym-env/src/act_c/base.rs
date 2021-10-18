use border_core::{Act, Shape};
use ndarray::ArrayD;
use std::{fmt::Debug, marker::PhantomData};

/// Represents an action.
#[derive(Clone, Debug)]
pub struct PyGymEnvContinuousAct<S: Shape> {
    /// Stores an action.
    pub act: ArrayD<f32>,
    pub(crate) phantom: PhantomData<S>,
}

impl<S: Shape> PyGymEnvContinuousAct<S> {
    /// Constructs an action.
    pub fn new(act: ArrayD<f32>) -> Self {
        Self {
            act,
            phantom: PhantomData,
        }
    }
}

impl<S: Shape> Act for PyGymEnvContinuousAct<S> {
    fn len(&self) -> usize {
        let shape = self.act.shape();
        if shape.len() == 1 {
            1
        } else {
            shape[0]
        }
    }
}
