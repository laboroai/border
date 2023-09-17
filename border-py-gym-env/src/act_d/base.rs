use border_core::Act;
use std::fmt::Debug;

/// Represents action.
#[derive(Clone, Debug)]
pub struct GymDiscreteAct {
    pub act: Vec<i32>,
}

impl GymDiscreteAct {
    /// Constructs a discrete action.
    pub fn new(act: Vec<i32>) -> Self {
        Self { act }
    }
}

impl Act for GymDiscreteAct {
    fn len(&self) -> usize {
        self.act.len()
    }
}
