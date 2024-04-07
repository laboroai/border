//! Exploration strategies of DQN.
use candle_core::Tensor;
use candle_nn::ops::softmax;
use ordered_float::OrderedFloat;
use rand::{
    distributions::{Uniform, WeightedIndex},
    Rng,
};
use serde::{Deserialize, Serialize};

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

    /// Takes an action based on action values, returns i64 tensor.
    ///
    /// * `a` - action values.
    pub fn action(&mut self, a: &Tensor, rng: &mut impl Rng) -> Tensor {
        let device = a.device();
        let probs = softmax(a, 1).unwrap().to_vec2::<f32>().unwrap();
        let n_samples = probs.len();
        let data = probs
            .into_iter()
            .map(|p| rng.sample(WeightedIndex::new(&p).unwrap()) as i64)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, &[n_samples], device).unwrap()
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

    /// Takes an action based on action values, returns i64 tensor.
    ///
    /// * `a` - action values.
    pub fn action(&mut self, a: &Tensor, rng: &mut impl Rng) -> Tensor {
        self.n_opts += 1;
        let device = a.device();
        let d = (self.eps_start - self.eps_final) / (self.final_step as f64);
        let eps =
            (10000.0 * (self.eps_start - d * self.n_opts as f64).max(self.eps_final)) as usize;
        let n_samples = a.dims()[0];
        let n_actions = a.dims()[1];
        let data = a
            .to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .map(|a| {
                let r = rng.sample(Uniform::new(0, 10000));
                match r < eps {
                    true => rng.sample(Uniform::new(0, n_actions)) as i64,
                    false => {
                        let a = a.into_iter().map(|v| OrderedFloat(v)).collect::<Vec<_>>();
                        (0..n_actions).max_by_key(|&i| &a[i]).unwrap() as i64
                    }
                }
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(data, &[n_samples], device).unwrap()
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
