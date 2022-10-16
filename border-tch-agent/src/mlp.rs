//! Multilayer perceptron.
mod base;
mod config;
mod mlp2;
pub use base::Mlp;
pub use config::MlpConfig;
pub use mlp2::Mlp2;
use tch::{nn};

fn mlp(prefix: &str, var_store: &nn::VarStore, config: &MlpConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    let mut in_dim = config.in_dim;
    let p = &var_store.root();

    for (i, &n) in config.units.iter().enumerate() {
        seq = seq.add(nn::linear(
            p / format!("{}{}", prefix, i + 1),
            in_dim,
            n,
            Default::default(),
        ));
        seq = seq.add_fn(|x| x.relu());
        in_dim = n;
    }

    seq
}
