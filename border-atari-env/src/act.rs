//! Action for [BorderAtariEnv](crate::BorderAtariEnv)
use border_core::Act;

#[derive(Debug, Clone)]
/// Action for [BorderAtariEnv](crate::BorderAtariEnv)
pub struct BorderAtariAct {
    pub(crate) act: u8,
}

impl Act for BorderAtariAct {
    fn len(&self) -> usize {
        1
    }
}
