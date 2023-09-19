use crate::util::OutDim;
use serde::{Deserialize, Serialize};

fn default_skip_linear() -> bool {
    false
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [`Cnn`](super::Cnn).
///
/// If `skip_linear` is `true`, `out_dim` is not used.
pub struct CnnConfig {
    pub n_stack: i64,
    pub out_dim: i64,
    #[serde(default = "default_skip_linear")]
    pub skip_linear: bool,
}

impl CnnConfig {
    /// Constructs [`CnnConfig`]
    pub fn new(n_stack: i64, out_dim: i64) -> Self {
        Self { n_stack, out_dim, skip_linear: false }
    }

    pub fn skip_linear(mut self, skip_linear: bool) -> Self {
        self.skip_linear = skip_linear;
        self
    }
}

impl OutDim for CnnConfig {
    /// Gets output dimension.
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    /// Sets output dimension.
    fn set_out_dim(&mut self, v: i64) {
        self.out_dim = v;
    }
}
