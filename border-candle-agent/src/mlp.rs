//! Multilayer perceptron.
mod base;
mod config;
mod mlp2;
use crate::Activation;
pub use base::Mlp;
use candle_core::Tensor;
use candle_nn::{Linear, Module};
pub use config::MlpConfig;
pub use mlp2::Mlp2;

fn mlp_forward(xs: Tensor, layers: &Vec<Linear>, final_act: &Activation) -> Tensor {
    let n_layers = layers.len();
    let mut xs = xs;

    for i in 0..=n_layers - 2 {
        xs = layers[i].forward(&xs).unwrap().relu().unwrap();
    }

    let xs = layers[n_layers - 1].forward(&xs).unwrap();
    final_act.forward(&xs)
}
