use super::{mlp_forward, MlpConfig};
use crate::model::SubModel1;
use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// Returns vector of linear modules from [`MlpConfig`].
fn create_linear_layers(prefix: &str, vs: VarBuilder, config: &MlpConfig) -> Result<Vec<Linear>> {
    let mut in_out_pairs: Vec<(i64, i64)> = (0..config.units.len() - 1)
        .map(|i| (config.units[i], config.units[i + 1]))
        .collect();
    in_out_pairs.insert(0, (config.in_dim, config.units[0]));
    let vs = vs.pp(prefix);

    Ok(in_out_pairs
        .iter()
        .enumerate()
        .map(|(i, &(in_dim, out_dim))| {
            linear(in_dim as _, out_dim as _, vs.pp(format!("ln{}", i))).unwrap()
        })
        .collect())
}

/// Multilayer perceptron that outputs two tensors of the same size.
pub struct Mlp2 {
    _config: MlpConfig,
    device: Device,
    head1: Linear,
    head2: Linear,
    layers: Vec<Linear>,
}

impl SubModel1 for Mlp2 {
    type Config = MlpConfig;
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, xs: &Self::Input) -> Self::Output {
        let xs = xs.to_device(&self.device).unwrap();
        let xs = mlp_forward(xs, &self.layers, &crate::Activation::ReLU);
        let mean = self.head1.forward(&xs).unwrap();
        let std = self.head2.forward(&xs).unwrap().exp().unwrap();
        (mean, std)
    }

    fn build(vs: VarBuilder, config: Self::Config) -> Self {
        let device = vs.device().clone();
        let layers = create_linear_layers("mlp", vs.clone(), &config).unwrap();
        let (head1, head2) = {
            let in_dim = *config.units.last().unwrap();
            let out_dim = config.out_dim;
            let head1 = linear(in_dim as _, out_dim as _, vs.pp(format!("mean"))).unwrap();
            let head2 = linear(in_dim as _, out_dim as _, vs.pp(format!("std"))).unwrap();
            (head1, head2)
        };

        Self {
            _config: config,
            device,
            head1,
            head2,
            layers,
        }
    }
}
