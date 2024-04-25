//! Exploration strategies of DQN.
use candle_core::{shape::D, DType, Tensor};
use candle_nn::ops::softmax;
use rand::{distributions::WeightedIndex, Rng};
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
        let d = (self.eps_start - self.eps_final) / (self.final_step as f64);
        let eps = (self.eps_start - d * self.n_opts as f64).max(self.eps_final);
        let r = rng.gen::<f32>();
        let is_random = r < eps as f32;
        self.n_opts += 1;

        if is_random {
            let n_samples = a.dims()[0];
            let n_actions = a.dims()[1] as u64;
            Tensor::from_slice(
                (0..n_samples)
                    .map(|_| (rng.gen::<u64>() % n_actions) as i64)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &[n_samples],
                a.device(),
            )
            .unwrap()
        } else {
            a.argmax(D::Minus1).unwrap().to_dtype(DType::I64).unwrap()
        }
    }

    /// Takes an action based on action values, returns i64 tensor.
    ///
    /// * `a` - action values.
    pub fn action_with_best(&mut self, a: &Tensor, rng: &mut impl Rng) -> (Tensor, bool) {
        let d = (self.eps_start - self.eps_final) / (self.final_step as f64);
        let eps = (self.eps_start - d * self.n_opts as f64).max(self.eps_final);
        let r = rng.gen::<f32>();
        let is_random = r < eps as f32;
        self.n_opts += 1;

        let best = a.argmax(D::Minus1).unwrap().to_dtype(DType::I64).unwrap();

        if is_random {
            let n_samples = a.dims()[0];
            let n_actions = a.dims()[1] as u64;
            let act = Tensor::from_slice(
                (0..n_samples)
                    .map(|_| (rng.gen::<u64>() % n_actions) as i64)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &[n_samples],
                a.device(),
            )
            .unwrap();
            let act_: Vec<i64> = act.to_vec1().unwrap();
            let best_: Vec<i64> = best.to_vec1().unwrap();
            (act, act_ == best_)
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
