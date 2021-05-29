//! Utilities for using multilayer perceptron.
use crate::agent::tch::{
    util::OutDim,
    model::{SubModel, SubModel2},
    sac::{Actor, ActorBuilder, Critic, CriticBuilder},
    opt::OptimizerConfig
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tch::{Tensor, nn, nn::Module, Device};

#[allow(unused_imports)]
use crate::agent::tch::sac::SAC;

#[allow(clippy::clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
/// Configuration of [MLP] and [MLP2].
pub struct MLPConfig {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
}

impl MLPConfig {
    fn new(in_dim: i64, units: Vec<i64>, out_dim: i64) -> Self {
        Self {
            in_dim, units, out_dim
        }
    }
}

impl OutDim for MLPConfig {
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    fn set_out_dim(&mut self, out_dim: i64) {
        self.out_dim = out_dim;
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
/// Multilayer perceptron, outputting two tensors.
pub struct MLP {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
    device: Device,
    seq: nn::Sequential,
}

fn mlp(var_store: &nn::VarStore, config: &MLPConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    let mut in_dim = config.in_dim;
    let p = &var_store.root();

    for (i, &n) in config.units.iter().enumerate() {
        seq = seq.add(nn::linear(p / format!("l{}", i), in_dim, n, Default::default()));
        seq = seq.add_fn(|x| x.relu());
        in_dim = n;
    }

    seq
}

impl SubModel2 for MLP {
    type Config = MLPConfig;
    type Input1 = Tensor;
    type Input2 = Tensor;
    type Output = Tensor;

    fn forward(&self, input1: &Self::Input1, input2: &Self::Input2) -> Self::Output {
        let input = Tensor::cat(&[input1, input2], -1).to(self.device);
        self.seq.forward(&input.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let units = &config.units;
        let in_dim = *units.last().unwrap_or(&config.in_dim);
        let out_dim = config.out_dim;
        let p = &var_store.root();
        let seq = mlp(var_store, &config)
            .add(nn::linear(p / format!("l{}", units.len()), in_dim, out_dim, Default::default()));

        Self {
            in_dim: config.in_dim,
            units: config.units,
            out_dim: config.out_dim,
            device: var_store.device(),
            seq
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        Self::build(var_store, Self::Config {
            in_dim: self.in_dim,
            units: self.units.clone(),
            out_dim: self.out_dim,
        })
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
/// Multilayer perceptron, taking two tensors.
pub struct MLP2 {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
    device: Device,
    head1: nn::Linear,
    head2: nn::Linear,
    seq: nn::Sequential,
}

impl SubModel for MLP2 {
    type Config = MLPConfig;
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let x = self.seq.forward(&input.to(self.device));
        let mean = x.apply(&self.head1);
        let std = x.apply(&self.head2).exp();
        (mean, std)
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let units = config.units;
        let out_dim = config.out_dim;
        let device = var_store.device();
        let mut seq = nn::seq();
        let mut in_dim = config.in_dim;
        let p = &var_store.root();

        for (i, &n) in units.iter().enumerate() {
            seq = seq.add(nn::linear(p / format!("l{}", i), in_dim, n, Default::default()));
            seq = seq.add_fn(|x| x.relu());
            in_dim = n;
        }

        let head1 = nn::linear(p / "head1", in_dim, out_dim as _, Default::default());
        let head2 = nn::linear(p / "head2", in_dim, out_dim as _, Default::default());

        let in_dim = config.in_dim;

        Self {
            in_dim,
            units,
            out_dim,
            device,
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
        };

        Self::build(var_store, config)
    }
}

/// Create an actor for [SAC]. It is represented by [MLP].
pub fn create_actor(in_dim: i64, out_dim: i64, lr_actor: f64, units: Vec<i64>, device: Device) -> Result<Actor<MLP2>> {
    ActorBuilder::default()
        .pi_config(MLPConfig::new(in_dim, units, out_dim))
        .opt_config(OptimizerConfig::Adam { lr: lr_actor})
        .build(device)
}

/// Create a cricit for [SAC]. It is represented by [MLP2].
pub fn create_critic(in_dim: i64, out_dim: i64, lr_critic: f64, units: Vec<i64>, device: Device) -> Result<Critic<MLP>> {
    CriticBuilder::default()
        .q_config(MLPConfig::new(in_dim, units, out_dim))
        .opt_config(OptimizerConfig::Adam { lr: lr_critic})
        .build(device)
}
