//! IQN model.
use std::{path::Path, default::Default, marker::PhantomData, error::Error};
use log::{info, trace};
use tch::{Tensor, Kind::Float, Device, nn, nn::OptimizerConfig};
use super::super::{
    util::{FeatureExtractor, FeatureExtractorBuilder},
    model::ModelBase
};

#[allow(clippy::upper_case_acronyms)]
/// Constructs [IQNModel].
///
/// The type parameter `F` is [FeatureExtractor].
pub struct IQNBuilder<F: FeatureExtractor> {
    phantom: PhantomData<F>
}

impl<F: FeatureExtractor> IQNBuilder<F> {
    /// Constructs [IQNModel].
    pub fn build<B>(builder: B, in_dim: i64, embed_dim: i64, out_dim: i64, learning_rate: f64, device: Device)
        -> IQNModel<F> where
        B: FeatureExtractorBuilder<F = F>,
    {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();
        let psi = builder.build(p);
        let lin1 = nn::linear(p, embed_dim, in_dim, Default::default());
        let lin2 = nn::linear(p, in_dim, out_dim, Default::default());
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();

        IQNModel {
            device,
            var_store: vs,
            in_dim,
            embed_dim,
            out_dim,
            lin1,
            lin2,
            psi,
            opt,
            phantom: PhantomData
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
/// Constructs IQN output layer, which takes input features and percent points.
/// It returns action-value quantiles.
///
/// The type parameter `F` is [FeatureExtractor].
pub struct IQNModel<F: FeatureExtractor> {
    device: Device,
    var_store: nn::VarStore,

    // Dimension of the input (feature) vector.
    in_dim: i64,

    // Dimension of the quantile embedding vector.
    embed_dim: i64,

    // Dimension of the output vector (equal to the number of actions).
    pub(super) out_dim: i64,

    // Linear layers
    lin1: nn::Linear,
    lin2: nn::Linear,

    // Feature extractor
    psi: F,

    // Optimizer
    opt: nn::Optimizer<nn::Adam>,

    phantom: PhantomData<F>
}

impl<F: FeatureExtractor> IQNModel<F> {
    /// Returns the tensor of action-value quantiles.
    ///
    /// * The shape of `tau` is [n_quantiles].
    /// * The shape of the output is [batch_size, n_quantiles, self.out_dim].
    pub fn forward(&self, x: &F::Input, tau: &Tensor) -> Tensor {
        // Feature extraction
        let psi = self.psi.feature(x);

        // * The shape of `psi` is [batch_size, self.in_dim].
        let batch_size = psi.size().as_slice()[0];
        let n_quantiles = tau.size().as_slice()[0];
        debug_assert_eq!(psi.size().as_slice(), &[batch_size, self.in_dim]);
        debug_assert_eq!(psi.size().as_slice()[1], self.in_dim);
        debug_assert_eq!(tau.size().as_slice(), &[n_quantiles]);

        // Eq. (4) in the paper
        let pi = std::f64::consts::PI;
        let i = Tensor::range(0, self.embed_dim - 1, (Float, self.device));
        let cos = Tensor::cos(&(tau.unsqueeze(-1) * ((pi * i).unsqueeze(0))));
        debug_assert_eq!(cos.size().as_slice(), &[n_quantiles, self.embed_dim]);
        let phi = cos.apply(&self.lin1).relu().unsqueeze(0);
        debug_assert_eq!(phi.size().as_slice(), &[1, n_quantiles, self.in_dim]);

        // Merge features and embedded quantiles by elem-wise multiplication
        let psi = psi.unsqueeze(1);
        debug_assert_eq!(psi.size().as_slice(), &[batch_size, 1, self.in_dim]);
        let m = psi * phi;
        debug_assert_eq!(m.size().as_slice(), &[batch_size, n_quantiles, self.in_dim]);

        // Action-value
        let a = m.apply(&self.lin2);
        debug_assert_eq!(a.size().as_slice(), &[batch_size, n_quantiles, self.out_dim]);

        a
    }
}

// TODO: implement this trait
impl<F: FeatureExtractor> ModelBase for IQNModel<F> {
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save IQN model to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        };
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load IQN model from {:?}", path.as_ref());
        Ok(())
    }
}

#[allow(clippy::upper_case_acronyms)]
/// The way of taking percent points.
pub enum IQNAverage {
    /// Average over percent points `0.05:0.1:0.95`.
    ///
    /// The precent points are constants.
    Uniform10
}

/// Takes an average over percent points specified by `mode`.
///
/// * `obs` - Observations.
/// * `iqn` - IQN model.
/// * `mode` - The way of taking percent points.
pub(super) fn average<F>(obs: &F::Input, iqn: &IQNModel<F>, mode: IQNAverage, device: Device)
    -> Tensor where
    F: FeatureExtractor
{
    use IQNAverage::*;

    let tau = match mode {
        Uniform10 => Tensor::of_slice(
            &[0.05_f32, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        ).to(device)
    };

    let averaged_action_value = iqn.forward(obs, &tau).mean1(&[1], false, Float);
    let batch_size = averaged_action_value.size()[0];
    let n_action = iqn.out_dim;
    debug_assert_eq!(averaged_action_value.size().as_slice(), &[batch_size, n_action]);
    averaged_action_value
}

#[test]
/// Check shape of tensors in IQNModel.
fn test_iqn_model() {
    let in_dim = 1000;
    let embed_dim = 64;
    let out_dim = 16;
    let n_quantiles = 8;
    let batch_size = 32;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let device = vs.device();
    let model = IQNModel::new(&vs.root(), in_dim, embed_dim, out_dim, device);
    let psi = Tensor::rand(&[batch_size, in_dim], tch::kind::FLOAT_CPU);
    let tau = Tensor::rand(&[n_quantiles], tch::kind::FLOAT_CPU);
    assert_eq!(tau.size().as_slice(), &[n_quantiles]);
    let _q = model.forward(&psi, &tau);
}
