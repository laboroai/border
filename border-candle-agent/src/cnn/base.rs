use super::CnnConfig;
use crate::model::SubModel1;
use anyhow::Result;
use candle_core::{DType::F32, Device, Tensor};
use candle_nn::{
    conv::Conv2dConfig,
    conv2d_no_bias, linear,
    sequential::{seq, Sequential},
    Module, VarBuilder,
};

#[allow(clippy::upper_case_acronyms)]
#[allow(dead_code)]
/// Convolutional neural network, which has the same architecture of the DQN paper.
pub struct Cnn {
    n_stack: i64,
    out_dim: i64,
    device: Device,
    seq: Sequential,
    skip_linear: bool,
}

impl Cnn {
    fn stride(s: i64) -> Conv2dConfig {
        Conv2dConfig {
            stride: s as _,
            ..Default::default()
        }
    }

    fn create_net(vb: &VarBuilder, n_stack: i64, out_dim: i64) -> Result<Sequential> {
        let seq = seq()
            .add_fn(|xs| xs.squeeze(2)?.to_dtype(F32)? / 255.0)
            .add(conv2d_no_bias(
                n_stack as _,
                32,
                8,
                Self::stride(4),
                vb.pp("c1"),
            )?)
            .add_fn(|xs| xs.relu())
            .add(conv2d_no_bias(32, 64, 4, Self::stride(2), vb.pp("c2"))?)
            .add_fn(|xs| xs.relu())
            .add(conv2d_no_bias(64, 64, 3, Self::stride(1), vb.pp("c3"))?)
            .add_fn(|xs| xs.relu()?.flatten_from(1))
            .add(linear(3136, 512, vb.pp("l1"))?)
            .add_fn(|xs| xs.relu())
            .add(linear(512, out_dim as _, vb.pp("l2"))?);

        Ok(seq)
    }

    fn create_net_wo_linear(vb: &VarBuilder, n_stack: i64) -> Result<Sequential> {
        let seq = seq()
            .add_fn(|xs| xs.squeeze(2)?.to_dtype(F32)? / 255.0)
            .add(conv2d_no_bias(
                n_stack as _,
                32,
                8,
                Self::stride(4),
                vb.pp("c1"),
            )?)
            .add_fn(|xs| xs.relu())
            .add(conv2d_no_bias(32, 64, 4, Self::stride(2), vb.pp("c2"))?)
            .add_fn(|xs| xs.relu())
            .add(conv2d_no_bias(64, 64, 3, Self::stride(1), vb.pp("c3"))?)
            .add_fn(|xs| xs.relu()?.flatten_from(1));

        Ok(seq)
    }
}

impl SubModel1 for Cnn {
    type Config = CnnConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Tensor {
        self.seq
            .forward(&x.to_device(&self.device).unwrap())
            .unwrap()
    }

    fn build(vb: VarBuilder, config: Self::Config) -> Self {
        let n_stack = config.n_stack;
        let out_dim = config.out_dim;
        let device = vb.device().clone();
        let skip_linear = config.skip_linear;
        let seq = if config.skip_linear {
            Self::create_net_wo_linear(&vb, n_stack)
        } else {
            Self::create_net(&vb, n_stack, out_dim)
        }
        .unwrap();

        Self {
            n_stack,
            out_dim,
            device,
            seq,
            skip_linear,
        }
    }
}
