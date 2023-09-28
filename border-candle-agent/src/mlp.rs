//! Multilayer perceptron.
mod base;
mod config;
mod mlp2;
pub use base::Mlp;
use candle_core::Tensor;
use candle_nn::{Linear, Module};
pub use config::MlpConfig;
pub use mlp2::Mlp2;

fn mlp_forward(xs: Tensor, layers: &Vec<Linear>) -> Tensor {
    let n_layers = layers.len();
    let mut xs = xs;

    for i in 0..=n_layers - 2 {
        xs = layers[i].forward(&xs).unwrap().relu().unwrap();
    }

    layers[n_layers - 1].forward(&xs).unwrap()
}

//     for (i, &n) in config.units.iter().enumerate() {
//         seq = seq.add(nn::linear(
//             p / format!("{}{}", prefix, i + 1),
//             in_dim,
//             n,
//             Default::default(),
//         ));
//         seq = seq.add_fn(|x| x.relu());
//         in_dim = n;
//     }

// }

// fn mlp(prefix: &str, var_store: &nn::VarStore, config: &MlpConfig) -> nn::Sequential {
//     let mut seq = nn::seq();
//     let mut in_dim = config.in_dim;
//     let p = &var_store.root();

//     for (i, &n) in config.units.iter().enumerate() {
//         seq = seq.add(nn::linear(
//             p / format!("{}{}", prefix, i + 1),
//             in_dim,
//             n,
//             Default::default(),
//         ));
//         seq = seq.add_fn(|x| x.relu());
//         in_dim = n;
//     }

//     seq
// }
