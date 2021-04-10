//! IQN model.
use std::{marker::PhantomData, default::Default};
use tch::{Tensor, Kind::Float, Device, nn};
// use crate::agent::tch::model::ModelBase;

/// Constructs IQN output layer.
pub struct IQNModel {
    device: Device,
    // Dimension of the input (feature) vector.
    in_dim: i64,
    // Dimension of the quantile embedding vector.
    embed_dim: i64,
    // Dimension of the output vector (equal to the number of actions).
    out_dim: i64,
    // // Feature extractor
    // psi: Psi,
    // Linear layers
    lin1: nn::Linear,
    lin2: nn::Linear,

    // phantom: PhantomData<Input>
}

impl IQNModel {
    /// Constructs IQN model.
    pub fn new(p: &nn::Path, in_dim: i64, embed_dim: i64, out_dim: i64, device: Device) -> Self {
        let lin1 = nn::linear(p, embed_dim, in_dim, Default::default());
        let lin2 = nn::linear(p, in_dim, out_dim, Default::default());

        Self {
            in_dim,
            embed_dim,
            out_dim,
            lin1,
            lin2,
            device,
            // phantom: PhantomData
        }
    }

    /// Returns the tensor of action-value quantiles.
    pub fn forward(&self, psi: &Tensor, tau: &Tensor) -> Tensor {
        let batch_size = psi.size().as_slice()[0];
        let n_quantiles = tau.size().as_slice()[0];
        debug_assert_eq!(psi.size().len(), 2);
        debug_assert_eq!(psi.size().as_slice()[1], self.in_dim);
        debug_assert_eq!(tau.size().len(), 1);

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

#[test]
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
