use super::{mlp_forward, MlpConfig};
use crate::model::SubModel1;
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{init::Init, linear, Linear, VarBuilder};

/// Returns vector of linear modules from [`MlpConfig`].
fn create_linear_layers(prefix: &str, vs: VarBuilder, config: &MlpConfig) -> Result<Vec<Linear>> {
    let mut in_out_pairs: Vec<(i64, i64)> = (0..config.units.len() - 1)
        .map(|i| (config.units[i], config.units[i + 1]))
        .collect();
    in_out_pairs.insert(0, (config.in_dim, config.units[0]));
    in_out_pairs.push((*config.units.last().unwrap(), config.out_dim));
    let vs = vs.pp(prefix);

    Ok(in_out_pairs
        .iter()
        .enumerate()
        .map(|(i, &(in_dim, out_dim))| {
            linear(in_dim as _, out_dim as _, vs.pp(format!("ln{}", i))).unwrap()
        })
        .collect())
}

/// A module with two heads.
///
/// The one is MLP, while the other is just a set of parameters.
/// This is based on the code of a neural network used in AWAC of CORL:
/// <https://github.com/tinkoff-ai/CORL/blob/6afec90484bbf47dee05fdf525e26a3ebe028e9b/algorithms/offline/awac.py>
pub struct Mlp3 {
    _config: MlpConfig,
    device: Device,
    layers: Vec<Linear>,
    head2: Tensor,
}

impl SubModel1 for Mlp3 {
    type Config = MlpConfig;
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, xs: &Self::Input) -> Self::Output {
        let batch_size = xs.dims()[0];
        let xs = xs.to_device(&self.device).unwrap();
        let ys = mlp_forward(xs, &self.layers, &crate::Activation::None);
        let zs = self.head2.repeat((batch_size, 1)).unwrap();
        (ys, zs)
    }

    fn build(vs: VarBuilder, config: Self::Config) -> Self {
        let device = vs.device().clone();
        let head2 = vs
            .get_with_hints((1, config.out_dim as usize), "head2", Init::Const(0.))
            .unwrap()
            .to_device(&device)
            .unwrap();
        let layers = create_linear_layers("mlp", vs, &config).unwrap();

        Mlp3 {
            _config: config,
            device,
            layers,
            head2,
        }
    }
}
