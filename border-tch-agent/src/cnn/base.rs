use super::AtariCnnConfig;
use crate::model::SubModel;
use tch::{nn, nn::Module, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
/// Convolutional neural network for Atari games, which has the same architecture of the DQN paper.
pub struct AtariCnn {
    n_stack: i64,
    out_dim: i64,
    device: Device,
    seq: nn::Sequential,
    skip_linear: bool,
}

impl AtariCnn {
    fn stride(s: i64) -> nn::ConvConfig {
        nn::ConvConfig {
            stride: s,
            ..Default::default()
        }
    }

    fn create_net(var_store: &nn::VarStore, n_stack: i64, out_dim: i64) -> nn::Sequential {
        let p = &var_store.root();
        nn::seq()
            .add_fn(|xs| xs.squeeze_dim(2).internal_cast_float(true) / 255)
            .add(nn::conv2d(p / "c1", n_stack, 32, 8, Self::stride(4)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, Self::stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, Self::stride(1)))
            .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 3136, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "l2", 512, out_dim as _, Default::default()))
    }

    fn create_net_wo_linear(var_store: &nn::VarStore, n_stack: i64) -> nn::Sequential {
        let p = &var_store.root();
        nn::seq()
            .add_fn(|xs| xs.squeeze_dim(2).internal_cast_float(true) / 255)
            .add(nn::conv2d(p / "c1", n_stack, 32, 8, Self::stride(4)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, Self::stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, Self::stride(1)))
            .add_fn(|xs| xs.relu().flat_view())
    }
}

impl SubModel for AtariCnn {
    type Config = AtariCnnConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Tensor {
        self.seq.forward(&x.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let n_stack = config.n_stack;
        let out_dim = config.out_dim;
        let device = var_store.device();
        let skip_linear = config.skip_linear;
        let seq = if config.skip_linear {
            Self::create_net_wo_linear(var_store, n_stack)
        } else {
            Self::create_net(var_store, n_stack, out_dim)
        };

        Self {
            n_stack,
            out_dim,
            device,
            seq,
            skip_linear,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let n_stack = self.n_stack;
        let out_dim = self.out_dim;
        let skip_linear = self.skip_linear;
        let device = var_store.device();
        let seq = if skip_linear {
            Self::create_net_wo_linear(&var_store, n_stack)
        } else {
            Self::create_net(&var_store, n_stack, out_dim)
        };

        Self {
            n_stack,
            out_dim,
            device,
            seq,
            skip_linear,
        }
    }
}
