//! Utilities used by agents.
use serde::{Deserialize, Serialize};

/// Interval between optimization steps.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub enum OptInterval {
    /// Optimization interval specified as interaction steps.
    Steps(usize),
    /// Optimization interval specified as episodes.
    Episodes(usize),
}

impl OptInterval {
    /// Constructs the counter for optimization.
    pub fn counter(self) -> OptIntervalCounter {
        OptIntervalCounter {
            opt_interval: self,
            count: 0,
        }
    }
}

/// The counter for optimization.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct OptIntervalCounter {
    opt_interval: OptInterval,
    count: usize,
}

impl OptIntervalCounter {
    /// Returns true if the optimization should be done.
    pub fn do_optimize(&mut self, is_done: &[i8]) -> bool {
        let is_done_any = is_done.iter().fold(0, |x, v| x + *v as i32) > 0;
        match self.opt_interval {
            OptInterval::Steps(interval) => {
                self.count += 1;
                if self.count == interval {
                    self.count = 0;
                    true
                } else {
                    false
                }
            }
            OptInterval::Episodes(interval) => {
                if is_done_any {
                    self.count += 1;
                    if self.count == interval {
                        self.count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}

/// Critic loss type.
#[allow(clippy::upper_case_acronyms)]
pub enum CriticLoss {
    /// Mean squared error.
    MSE,

    /// Smooth L1 loss.
    SmoothL1,
}
