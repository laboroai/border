use border_core::Act;
use std::fmt::Debug;

/// Represents action.
#[derive(Clone, Debug)]
pub struct PyGymEnvDiscreteAct {
    pub act: Vec<i32>,
}

impl PyGymEnvDiscreteAct {
    /// Constructs a discrete action.
    pub fn new(act: Vec<i32>) -> Self {
        Self { act }
    }
}

impl Act for PyGymEnvDiscreteAct {
    fn len(&self) -> usize {
        self.act.len()
    }
}
