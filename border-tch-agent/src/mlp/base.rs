use super::{mlp, MlpConfig};
use crate::model::{SubModel, SubModel2};
use tch::{nn, nn::Module, Device, Tensor};

/// Multilayer perceptron with ReLU activation function.
pub struct Mlp {
    config: MlpConfig,
    device: Device,
    seq: nn::Sequential,
}

impl Mlp {
    fn create_net(var_store: &nn::VarStore, config: &MlpConfig) -> nn::Sequential {
        let p = &(var_store.root() / "mlp");
        let mut seq = nn::seq();
        let mut in_dim = config.in_dim;

        for (i, &out_dim) in config.units.iter().enumerate() {
            seq = seq.add(nn::linear(
                p / format!("{}{}", "ln", i),
                in_dim,
                out_dim,
                Default::default(),
            ));
            seq = seq.add_fn(|x| x.relu());
            in_dim = out_dim;
        }

        seq = seq.add(nn::linear(
            p / format!("{}{}", "ln", config.units.len()),
            in_dim,
            config.out_dim,
            Default::default(),
        ));

        if config.activation_out {
            seq = seq.add_fn(|x| x.relu());
        }

        seq
    }
}

impl SubModel for Mlp {
    type Config = MlpConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Tensor {
        self.seq.forward(&x.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let device = var_store.device();
        let seq = Self::create_net(var_store, &config);

        Self {
            config,
            device,
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let config = self.config.clone();
        let device = var_store.device();
        let seq = Self::create_net(&var_store, &config);

        Self {
            config,
            device,
            seq,
        }
    }
}

impl SubModel2 for Mlp {
    type Config = MlpConfig;
    type Input1 = Tensor;
    type Input2 = Tensor;
    type Output = Tensor;

    fn forward(&self, input1: &Self::Input1, input2: &Self::Input2) -> Self::Output {
        let input1: Tensor = input1.to(self.device);
        let input2: Tensor = input2.to(self.device);
        let input = Tensor::cat(&[input1, input2], -1);
        self.seq.forward(&input.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let units = &config.units;
        let in_dim = *units.last().unwrap_or(&config.in_dim);
        let out_dim = config.out_dim;
        let p = &(var_store.root() / "mlp");
        let seq = mlp("ln", var_store, &config).add(nn::linear(
            p / format!("ln{}", units.len()),
            in_dim,
            out_dim,
            Default::default(),
        ));

        Self {
            config,
            device: var_store.device(),
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let config = self.config.clone();
        let device = var_store.device();
        let seq = Self::create_net(&var_store, &config);

        Self {
            config,
            device,
            seq,
        }
    }
}
