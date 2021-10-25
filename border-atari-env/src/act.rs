//! Action for [BorderAtariEnv](crate::BorderAtariEnv)
use border_core::Act;

#[derive(Debug, Clone)]
/// Action for [BorderAtariEnv](crate::BorderAtariEnv)
pub struct BorderAtariAct {
    pub(crate) act: u8,
}

impl BorderAtariAct {
    pub fn new(act: u8) -> Self {
        Self { act }
    }
}

impl Act for BorderAtariAct {
    fn len(&self) -> usize {
        1
    }
}

impl From<u8> for BorderAtariAct {
    fn from(act: u8) -> Self {
        Self { act }
    }
}
