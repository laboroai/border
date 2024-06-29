//! Exploration strategies of DQN.
use std::convert::TryInto;

use serde::{Deserialize, Serialize};
use tch::Tensor;

/// Explorers for DQN.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum DqnExplorer {
    /// Softmax action selection.
    Softmax(Softmax),

    /// Epsilon-greedy action selection.
    EpsilonGreedy(EpsilonGreedy),
}

/// Softmax explorer for DQN.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct Softmax {}

#[allow(clippy::new_without_default)]
impl Softmax {
    /// Constructs softmax explorer.
    pub fn new() -> Self {
        Self {}
    }

    /// Takes an action based on the observation and the critic.
    pub fn action(&mut self, a: &Tensor) -> Tensor {
        a.softmax(-1, tch::Kind::Float).multinomial(1, true)
    }
}

/// Epsilon-greedy explorer for DQN.
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct EpsilonGreedy {
    pub n_opts: usize,
    pub eps_start: f64,
    pub eps_final: f64,
    pub final_step: usize,
}

#[allow(clippy::new_without_default)]
impl EpsilonGreedy {
    /// Constructs epsilon-greedy explorer.
    pub fn new() -> Self {
        Self {
            n_opts: 0,
            eps_start: 1.0,
            eps_final: 0.02,
            final_step: 100_000,
        }
    }

    /// Constructs epsilon-greedy explorer.
    ///
    /// TODO: improve interface.
    pub fn with_final_step(final_step: usize) -> DqnExplorer {
        DqnExplorer::EpsilonGreedy(Self {
            n_opts: 0,
            eps_start: 1.0,
            eps_final: 0.02,
            final_step,
        })
    }

    /// Takes an action based on the observation and the critic.
    pub fn action(&mut self, a: &Tensor) -> Tensor {
        let d = (self.eps_start - self.eps_final) / (self.final_step as f64);
        let eps = (self.eps_start - d * self.n_opts as f64).max(self.eps_final);
        let r = fastrand::f64();
        let is_random = r < eps;
        self.n_opts += 1;

        let best = a.argmax(-1, true);

        if is_random {
            let n_procs = a.size()[0] as u32;
            let n_actions = a.size()[1] as u32;
            let act = Tensor::from_slice(
                (0..n_procs)
                    .map(|_| fastrand::u32(..n_actions) as i32)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            act
        } else {
            best
        }
    }

    /// Takes an action based on the observation and the critic.
    pub fn action_with_best(&mut self, a: &Tensor) -> (Tensor, bool) {
        let d = (self.eps_start - self.eps_final) / (self.final_step as f64);
        let eps = (self.eps_start - d * self.n_opts as f64).max(self.eps_final);
        let r = fastrand::f64();
        let is_random = r < eps;
        self.n_opts += 1;

        let best = a.argmax(-1, true);

        if is_random {
            let n_procs = a.size()[0] as u32;
            let n_actions = a.size()[1] as u32;
            let act = Tensor::from_slice(
                (0..n_procs)
                    .map(|_| fastrand::u32(..n_actions) as i32)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            let diff: i64 = (&act - &best.to(tch::Device::Cpu))
                .abs()
                .sum(tch::Kind::Int64)
                .try_into()
                .unwrap();
            (act, diff == 0)
        } else {
            (best, true)
        }
    }

    /// Set the epsilon value at the final step.
    pub fn eps_final(self, v: f64) -> Self {
        let mut s = self;
        s.eps_final = v;
        s
    }

    /// Set the epsilon value at the start.
    pub fn eps_start(self, v: f64) -> Self {
        let mut s = self;
        s.eps_start = v;
        s
    }
}
