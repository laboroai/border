/// Entropy coefficient of SAC.
use std::borrow::Borrow;
use env_logger::Target;
use tch::{Tensor, nn, nn::OptimizerConfig};

/// Mode of the entropy coefficient of SAC.
pub enum EntCoefMode {
    /// Use a constant as alpha.
    Fix(f64),
    /// Automatic tuning given `(target_entropy, learning_rate)`.
    Auto(f64, f64)
}

/// The entropy coefficient of SAC.
pub struct EntCoef {
    var_store: nn::VarStore,
    log_alpha: Tensor,
    target_entropy: Option<f64>,
    opt: Option<nn::Optimizer<nn::Adam>>,
}

impl EntCoef {
    pub fn new(mode: EntCoefMode, device: tch::Device) -> Self {
        let var_store = nn::VarStore::new(device);
        let path = &var_store.root();
        let (log_alpha, target_entropy, opt) = match mode {
            EntCoefMode::Fix(alpha) => {
                let init = nn::Init::Const(alpha.ln());
                let log_alpha = path.borrow().var("log_alpha", &[1], init);
                (log_alpha, None, None)
            },
            EntCoefMode::Auto(target_entropy, learning_rate) => {
                let init = nn::Init::Const(0.0);
                let log_alpha = path.borrow().var("log_alpha", &[1], init);
                let opt = nn::Adam::default().build(&var_store, learning_rate).unwrap();
                (log_alpha, Some(target_entropy), Some(opt))
            },
        };

        Self {
            var_store,
            log_alpha,
            opt,
            target_entropy,
        }
    }

    pub fn alpha(&self) -> Tensor {
        self.log_alpha.exp()
    }
}
