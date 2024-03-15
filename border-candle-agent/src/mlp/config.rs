use crate::util::OutDim;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`Mlp`](super::Mlp).
pub struct MlpConfig {
    pub(super) in_dim: i64,
    pub(super) units: Vec<i64>,
    pub(super) out_dim: i64,
    pub(super) activation_out: bool,
}

impl MlpConfig {
    /// Creates configuration of MLP.
    ///
    /// * `activation_out` - If `true`, activation function is added in the final layer.
    pub fn new(in_dim: i64, units: Vec<i64>, out_dim: i64, activation_out: bool) -> Self {
        Self {
            in_dim,
            units,
            out_dim,
            activation_out,
        }
    }
}

impl OutDim for MlpConfig {
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    fn set_out_dim(&mut self, out_dim: i64) {
        self.out_dim = out_dim;
    }
}
