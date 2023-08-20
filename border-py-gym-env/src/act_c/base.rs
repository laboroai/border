use border_core::Act;
use ndarray::ArrayD;
use std::fmt::Debug;

/// Represents an action.
#[derive(Clone, Debug)]
pub struct GymContinuousAct {
    /// Stores an action.
    pub act: ArrayD<f32>,
}

impl GymContinuousAct {
    /// Constructs an action.
    pub fn new(act: ArrayD<f32>) -> Self {
        Self {
            act,
        }
    }
}

impl Act for GymContinuousAct {
    fn len(&self) -> usize {
        let shape = self.act.shape();
        if shape.len() == 1 {
            1
        } else {
            shape[0]
        }
    }
}
