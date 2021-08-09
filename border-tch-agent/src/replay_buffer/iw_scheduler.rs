//! Scheduling the exponent of importance weight for PER.
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct IwScheduler {
    /// Initial value of $\beta$.
    pub beta_0: f32,

    /// Final value of $\beta$.
    pub beta_final: f32,

    /// Optimization steps when beta reaches its final value.
    pub n_opts_final: usize,
}

impl IwScheduler {
    pub fn beta(&self, n_opts: usize) -> f32 {
        if n_opts >= self.n_opts_final {
            self.beta_final
        } else {
            let d = self.beta_final - self.beta_0;
            self.beta_0 + d * (n_opts as f32 / self.n_opts_final as f32)
        }
    }
}
