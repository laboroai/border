use super::{mlp, MlpConfig};
use crate::model::SubModel;
use tch::{nn, nn::Module, Device, Tensor};

#[allow(clippy::clippy::upper_case_acronyms)]
/// Multilayer perceptron that outputs two tensors of the same size.
pub struct Mlp2 {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
    activation_out: bool,
    device: Device,
    head1: nn::Linear,
    head2: nn::Linear,
    seq: nn::Sequential,
}

impl SubModel for Mlp2 {
    type Config = MlpConfig;
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let x = self.seq.forward(&input.to(self.device));
        let mean = x.apply(&self.head1);
        let std = x.apply(&self.head2).exp();
        (mean, std)
    }

    /// TODO: support activation_out
    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let seq = mlp("al", var_store, &config);
        let out_dim = config.out_dim;
        let in_dim = *config.units.last().unwrap();
        let p = &var_store.root();

        let head1 = nn::linear(p / "ml", in_dim, out_dim as _, Default::default());
        let head2 = nn::linear(p / "sl", in_dim, out_dim as _, Default::default());

        Self {
            in_dim: config.in_dim,
            units: config.units,
            out_dim: config.out_dim,
            activation_out: false,
            device: var_store.device(),
            head1,
            head2,
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let config = Self::Config {
            in_dim: self.in_dim,
            units: self.units.clone(),
            out_dim: self.out_dim,
            activation_out: self.activation_out,
        };

        Self::build(var_store, config)
    }
}
