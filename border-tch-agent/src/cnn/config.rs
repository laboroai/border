use crate::util::OutDim;
use serde::{Deserialize, Serialize};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [CNN](super::CNN).
pub struct CNNConfig {
    pub(super) n_stack: i64,
    pub(super) out_dim: i64,
}

impl OutDim for CNNConfig {
    /// Gets output dimension.
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    /// Sets output dimension.
    fn set_out_dim(&mut self, v: i64) {
        self.out_dim = v;
    }
}
