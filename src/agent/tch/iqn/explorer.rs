//! Exploration strategies of IQN.
use std::default::Default;
use tch::Tensor;

#[allow(clippy::upper_case_acronyms)]
/// Explorers for IQN.
pub enum IQNExplorer
{
    // /// Softmax action selection.
    // Softmax(Softmax),

    /// Epsilon-greedy action selection.
    EpsilonGreedy(EpsilonGreedy)
}

// /// Softmax explorer for IQN.
// pub struct Softmax {}

// #[allow(clippy::new_without_default)]
// impl Softmax {
//     /// Constructs softmax explorer.
//     pub fn new() -> Self { Self {} }

//     /// Takes an action based on the observation and the critic.
//     pub fn action<M>(&mut self, qnet: &M, obs: &Tensor) -> Tensor where
//         M: Model1<Input=Tensor, Output=Tensor>,
//     {
//         let a = qnet.forward(obs);
//         a.softmax(-1, tch::Kind::Float).multinomial(1, true)
//     }
// }

/// Epsilon-greedy explorer for IQN.
pub struct EpsilonGreedy {
    n_opts: usize,
    eps_start: f64,
    eps_final: f64,
    final_step: usize,
}

impl Default for EpsilonGreedy {
    fn default() -> Self {
        Self {
            n_opts: 0,
            eps_start: 1.0,
            eps_final: 0.02,
            final_step: 100_000,
        }
    }
}

#[allow(clippy::new_without_default)]
impl EpsilonGreedy {
    /// Constructs epsilon-greedy explorer.
    pub fn with_params(eps_start: f64, eps_final: f64, final_step: usize) -> IQNExplorer {
        IQNExplorer::EpsilonGreedy(
            Self {
                n_opts: 0,
                eps_start,
                eps_final,
                final_step
            }
        )
    }

    /// Constructs epsilon-greedy explorer.
    ///
    /// TODO: improve interface.
    pub fn with_final_step(final_step: usize) -> IQNExplorer {
        IQNExplorer::EpsilonGreedy(
            Self {
                n_opts: 0,
                eps_start: 1.0,
                eps_final: 0.02,
                final_step,
            }
        )
    }

    /// Takes an action based on the observation and the critic.
    pub fn action<F: FnOnce() -> Tensor>(&mut self, shape: (u32, u32), q_fn: F) -> Tensor {
        let d = (self.eps_start - self.eps_final) / (self.final_step as f64);
        let eps = (self.eps_start - d * self.n_opts as f64).max(self.eps_final);
        let r = fastrand::f64();
        let is_random = r < eps;
        self.n_opts += 1;

        if is_random {
            let (n_procs, n_actions) = shape;
            Tensor::of_slice(
                (0..n_procs).map(|_| fastrand::u32(..n_actions) as i32).collect::<Vec<_>>()
                .as_slice()
            )
        }
        else {
            let a = q_fn();
            a.argmax(-1, true)
        }
    }
}
