//! Example of using IQNModel for quantile regression.
use border_tch_agent::{
    iqn::{IQNModel, IQNModelBuilder},
    model::{ModelBase, SubModel},
    util::quantile_huber_loss,
    util::OutDim,
};
use serde::{Deserialize, Serialize};
use std::default::Default;
use tch::{
    kind::FLOAT_CPU,
    nn,
    nn::{Module, VarStore},
    IndexOp, Tensor,
};

const N_SAMPLE: i64 = 300;
const N_TRAIN_STEP: i64 = 10000;
const BATCH_SIZE: i64 = 64;
const N_PERCENT_POINTS: i64 = 5;
const FEATURE_DIM: i64 = 8;
const EMBED_DIM: i64 = 64;

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct LinearConfig {
    in_dim: i64,
    out_dim: i64,
}

impl OutDim for LinearConfig {
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    fn set_out_dim(&mut self, v: i64) {
        self.out_dim = v;
    }
}

impl LinearConfig {
    pub fn new(in_dim: i64, out_dim: i64) -> Self {
        Self { in_dim, out_dim }
    }
}

struct Linear {
    in_dim: i64,
    out_dim: i64,
    linear: nn::Linear,
}

impl SubModel for Linear {
    type Config = LinearConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn build(var_store: &VarStore, config: Self::Config) -> Self {
        let p = &var_store.root();
        let in_dim = config.in_dim;
        let out_dim = config.out_dim;
        Linear {
            in_dim,
            out_dim,
            linear: nn::linear(p, in_dim, out_dim, Default::default()),
        }
    }

    fn clone_with_var_store(&self, var_store: &VarStore) -> Self {
        let p = &var_store.root();
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        Linear {
            in_dim,
            out_dim,
            linear: nn::linear(p, in_dim, out_dim, Default::default()),
        }
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        self.linear.forward(input)
    }
}

// Samples percent points
fn sample_percent_points() -> Tensor {
    Tensor::of_slice(&[0.1, 0.3, 0.5, 0.7, 0.9])
        .internal_cast_float(true)
        .unsqueeze(0)
        .repeat(&[BATCH_SIZE, 1])
}

// Returns pair (xs, ys) of inputs and outputs.
//
// Either of `xs.size()` and `ys.size()` are `(N_SAMPLE, 1)`.
fn create_data() -> (Tensor, Tensor) {
    let slope = 2.0;
    let bias = -3.0;
    let log_var_slope = 0.2;
    let xs: Tensor = 10.0 * Tensor::rand(&[N_SAMPLE, 1], FLOAT_CPU) - 5.0;
    let noise_scale = Tensor::exp(&(log_var_slope * &xs + 1.0));
    let eps = Tensor::zeros(&[N_SAMPLE, 1], FLOAT_CPU).normal_(0.0, 1.0) * noise_scale;
    let ys: Tensor = slope * &xs + bias + eps;
    (xs, ys)
}

// Creates IQNModel
fn create_iqn_model() -> IQNModel<Linear, Linear> {
    let fe_config = LinearConfig::new(1, FEATURE_DIM);
    let m_config = LinearConfig::new(FEATURE_DIM, 1);
    IQNModelBuilder::default()
        .feature_dim(FEATURE_DIM)
        .embed_dim(EMBED_DIM)
        .learning_rate(1e-4)
        .build_with_submodel_configs(fe_config, m_config, tch::Device::Cpu)
}

fn main() {
    // Constructs data and model
    let data = create_data();
    let mut model = create_iqn_model();

    // Trains the model
    for _ in 0..N_TRAIN_STEP {
        let (xs, ys) = Tensor::random_batch2(&data.0, &data.1, BATCH_SIZE);
        assert_eq!(xs.size().as_slice(), &[BATCH_SIZE, 1]);
        assert_eq!(ys.size().as_slice(), &[BATCH_SIZE, 1]);

        let tau = sample_percent_points();
        assert_eq!(tau.size().as_slice(), &[BATCH_SIZE, N_PERCENT_POINTS]);

        let pred = model.forward(&xs, &tau);
        assert_eq!(pred.size().as_slice(), &[BATCH_SIZE, N_PERCENT_POINTS, 1]);

        let ys = ys.unsqueeze(-1).repeat(&[1, N_PERCENT_POINTS, 1]);
        let tau = tau.unsqueeze(-1);
        let diff = ys - pred;
        let loss = quantile_huber_loss(&diff, &tau).mean(tch::Kind::Float);
        model.backward_step(&loss);
    }

    // Write data to file
    let (xs, ys) = data;
    let data = Tensor::stack(&[xs, ys], 1);
    let mut wtr = csv::Writer::from_path("examples/iqn_regression_data.csv").unwrap();
    (0..data.size()[0])
        .map(|i| data.i(i))
        .map(|t| {
            Vec::<f32>::from(&t)
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
        })
        .for_each(|v| wtr.write_record(&v).unwrap());

    // Write prediction to file
    let xs = (Tensor::range(0, 99, FLOAT_CPU) / 100.0 * 10.0 - 5.0).unsqueeze(-1);
    let tau = Tensor::of_slice(&[0.1f32, 0.3, 0.5, 0.7, 0.9])
        .unsqueeze(0)
        .repeat(&[100, 1]);
    let ys = model.forward(&xs, &tau).squeeze1(-1);
    let data = Tensor::cat(&[xs, ys], 1);
    assert_eq!(data.size().as_slice(), &[100, 6]);
    let mut wtr = csv::Writer::from_path("examples/iqn_regression_pred.csv").unwrap();
    (0..data.size()[0])
        .map(|i| data.i(i))
        .map(|t| {
            Vec::<f32>::from(&t)
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
        })
        .for_each(|v| wtr.write_record(&v).unwrap());
}
