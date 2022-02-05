use crate::util::OutDim;
use serde::{Deserialize, Serialize};

#[allow(clippy::clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [MLP](super::MLP).
pub struct MLPConfig {
    pub(super) in_dim: i64,
    pub(super) units: Vec<i64>,
    pub(super) out_dim: i64,
}

impl MLPConfig {
    pub fn new(in_dim: i64, units: Vec<i64>, out_dim: i64) -> Self {
        Self {
            in_dim,
            units,
            out_dim,
        }
    }
}

impl OutDim for MLPConfig {
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    fn set_out_dim(&mut self, out_dim: i64) {
        self.out_dim = out_dim;
    }
}
