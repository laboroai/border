use super::{mlp_forward, MlpConfig};
use crate::model::{SubModel1, SubModel2};
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

/// Returns vector of linear modules from [`MlpConfig`].
fn create_linear_layers(prefix: &str, vs: VarBuilder, config: &MlpConfig) -> Result<Vec<Linear>> {
    let mut in_out_pairs: Vec<(usize, usize)> = (0..config.units.len() - 1)
        .map(|i| (config.units[i], config.units[i + 1]))
        .collect();
    in_out_pairs.insert(0, (config.in_dim, config.units[0]));
    in_out_pairs.push((*config.units.last().unwrap(), config.out_dim));
    let vs = vs.pp(prefix);

    Ok(in_out_pairs
        .iter()
        .enumerate()
        .map(|(i, &(in_dim, out_dim))| linear(in_dim, out_dim, vs.pp(format!("ln{}", i))).unwrap())
        .collect())
}

/// Multilayer perceptron with ReLU activation function.
pub struct Mlp {
    config: MlpConfig,
    device: Device,
    layers: Vec<Linear>,
}

fn _build(vs: VarBuilder, config: MlpConfig) -> Mlp {
    let device = vs.device().clone();
    let layers = create_linear_layers("mlp", vs, &config).unwrap();

    Mlp {
        config,
        device,
        layers,
    }
}

impl SubModel1 for Mlp {
    type Config = MlpConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, xs: &Self::Input) -> Tensor {
        let xs = xs.to_device(&self.device).unwrap();
        let xs = mlp_forward(xs, &self.layers);

        match self.config.activation_out {
            false => xs,
            true => xs.relu().unwrap(),
        }
    }

    fn build(vs: VarBuilder, config: Self::Config) -> Self {
        _build(vs, config)
    }
}

impl SubModel2 for Mlp {
    type Config = MlpConfig;
    type Input1 = Tensor;
    type Input2 = Tensor;
    type Output = Tensor;

    fn forward(&self, input1: &Self::Input1, input2: &Self::Input2) -> Self::Output {
        let input1: Tensor = input1.to_device(&self.device).unwrap();
        let input2: Tensor = input2.to_device(&self.device).unwrap();
        let input = Tensor::cat(&[input1, input2], D::Minus1)
            .unwrap()
            .to_device(&self.device)
            .unwrap();
        let xs = mlp_forward(input, &self.layers);

        match self.config.activation_out {
            false => xs,
            true => xs.relu().unwrap(),
        }
    }

    fn build(vs: VarBuilder, config: Self::Config) -> Self {
        _build(vs, config)
    }
}
