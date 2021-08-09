use border_tch_agent::{model::SubModel, util::OutDim};
use serde::{Deserialize, Serialize};
use tch::{nn, nn::Module, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct CNNConfig {
    n_stack: i64,
    out_dim: i64,
}

impl OutDim for CNNConfig {
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    fn set_out_dim(&mut self, v: i64) {
        self.out_dim = v;
    }
}

impl CNNConfig {
    #[allow(dead_code)]
    pub fn new(n_stack: i64, out_dim: i64) -> Self {
        Self {
            n_stack,
            out_dim,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
// Convolutional neural network
pub struct CNN {
    n_stack: i64,
    out_dim: i64,
    device: Device,
    seq: nn::Sequential,
}

impl CNN {
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
}

impl SubModel for CNN {
    type Config = CNNConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Tensor {
        self.seq.forward(&x.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let n_stack = config.n_stack;
        let out_dim = config.out_dim;
        let device = var_store.device();
        let seq = Self::create_net(var_store, n_stack, out_dim);

        Self {
            n_stack,
            out_dim,
            device,
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let n_stack = self.n_stack;
        let out_dim = self.out_dim;
        let device = var_store.device();
        let seq = Self::create_net(&var_store, n_stack, out_dim);

        Self {
            n_stack,
            out_dim,
            device,
            seq,
        }
    }
}
