// use anyhow::Result;
// use border::{
//     agent::{
//         tch::{
//             iqn::{EpsilonGreedy, IQNBuilder},
//             ReplayBuffer,
//         },
//         OptInterval,
//     },
//     env::py_gym_env::{
//         act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
//         obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
//         tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
//         PyGymEnv, Shape,
//     },
//     shape,
// };
// use border_core::{
//     record::{BufferedRecorder, Record, TensorboardRecorder},
//     util, Agent, TrainerBuilder,
// };
// use clap::{App, Arg};
// use csv::WriterBuilder;
// use serde::Serialize;
// use std::{convert::TryFrom, default::Default, fs::File};

use anyhow::Result;
use border::try_from;
use border_core::{
    record::{BufferedRecorder, Record, TensorboardRecorder},
    shape, util, Agent, Shape, TrainerBuilder,
};
use border_py_gym_env::{newtype_act_d, newtype_obs, PyGymEnv, PyGymEnvDiscreteAct};
use border_tch_agent::{
    iqn::{EpsilonGreedy, IQNBuilder},
    replay_buffer::TchTensorBuffer,
    util::OptInterval,
};
use clap::{App, Arg};
use csv::WriterBuilder;
use serde::Serialize;
use std::{convert::TryFrom, fs::File};
use tch::Tensor;

const DIM_FEATURE: i64 = 256;
const DIM_EMBED: i64 = 64;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.1;
const SOFT_UPDATE_INTERVAL: usize = 100;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 10000;
const EVAL_INTERVAL: usize = 500;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const EPS_START: f64 = 1.0;
const EPS_FINAL: f64 = 0.1;
const FINAL_STEP: usize = 5000; // MAX_OPTS;
const MODEL_DIR: &str = "examples/model/iqn_cartpole";

shape!(ObsShape, [4]);
shape!(ActShape, [1]);
newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
newtype_act_d!(Act, ActFilter);

impl From<Obs> for Tensor {
    fn from(obs: Obs) -> Tensor {
        try_from(obs.0.obs).unwrap()
    }
}

impl From<Act> for Tensor {
    fn from(act: Act) -> Tensor {
        let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();

        // The first dimension of the action tensor is the number of processes,
        // which is 1 for the non-vectorized environment.
        t.unsqueeze(0)
    }
}

impl From<Tensor> for Act {
    /// `t` must be a 1-dimentional tensor of `f32`.
    fn from(t: Tensor) -> Self {
        let data: Vec<i64> = t.into();
        let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
        Act(PyGymEnvDiscreteAct::new(data))
    }
}

type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
type ActBuffer = TchTensorBuffer<i64, ActShape, Act>;

mod iqn_model {
    use border_tch_agent::{
        iqn::{IQNModel, IQNModelBuilder},
        model::SubModel,
        util::OutDim,
    };
    use serde::{Deserialize, Serialize};
    use tch::{nn, nn::Module, Device, Tensor};

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
    pub struct FCConfig {
        in_dim: i64,
        out_dim: i64,
        relu: bool,
    }

    impl OutDim for FCConfig {
        fn get_out_dim(&self) -> i64 {
            self.out_dim
        }

        fn set_out_dim(&mut self, v: i64) {
            self.out_dim = v;
        }
    }

    impl FCConfig {
        fn new(in_dim: i64, out_dim: i64, relu: bool) -> Self {
            Self {
                in_dim,
                out_dim,
                relu,
            }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    // Fully connected layer as feature extractor and network output
    pub struct FC {
        in_dim: i64,
        out_dim: i64,
        relu: bool,
        device: Device,
        seq: nn::Sequential,
    }

    impl FC {
        fn create_net(
            var_store: &nn::VarStore,
            in_dim: i64,
            out_dim: i64,
            relu: bool,
        ) -> nn::Sequential {
            let p = &var_store.root();
            let mut seq = nn::seq().add(nn::linear(p / "cl1", in_dim, out_dim, Default::default()));
            if relu {
                seq = seq.add_fn(|xs| xs.relu());
            }
            seq
        }
    }

    impl SubModel for FC {
        type Config = FCConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, x: &Self::Input) -> Tensor {
            self.seq.forward(&x.to(self.device))
        }

        fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
            let in_dim = config.in_dim;
            let out_dim = config.out_dim;
            let relu = config.relu;
            let device = var_store.device();
            let seq = Self::create_net(var_store, in_dim, out_dim, relu);

            Self {
                in_dim,
                out_dim,
                relu,
                device,
                seq,
            }
        }

        fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
            let in_dim = self.in_dim;
            let out_dim = self.out_dim;
            let relu = self.relu;
            let device = var_store.device();
            let seq = Self::create_net(&var_store, in_dim, out_dim, relu);

            Self {
                in_dim,
                out_dim,
                relu,
                device,
                seq,
            }
        }
    }

    // IQN model
    pub fn create_iqn_model(
        in_dim: i64,
        feature_dim: i64,
        embed_dim: i64,
        out_dim: i64,
        learning_rate: f64,
        device: Device,
    ) -> IQNModel<FC, FC> {
        let fe_config = FCConfig::new(in_dim, feature_dim, true);
        let m_config = FCConfig::new(feature_dim, out_dim, false);
        IQNModelBuilder::default()
            .feature_dim(feature_dim)
            .embed_dim(embed_dim)
            .learning_rate(learning_rate)
            .build_with_submodel_configs(fe_config, m_config, device)
    }
}

fn create_agent() -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let iqn_model = iqn_model::create_iqn_model(
        ObsShape::shape()[0] as _,
        DIM_FEATURE,
        DIM_EMBED,
        DIM_ACT,
        LR_CRITIC,
        device,
    );
    IQNBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .soft_update_interval(SOFT_UPDATE_INTERVAL)
        .explorer(EpsilonGreedy::with_params(EPS_START, EPS_FINAL, FINAL_STEP))
        .replay_buffer_capacity(REPLAY_BUFFER_CAPACITY)
        .build::<_, _, _, ObsBuffer, ActBuffer>(iqn_model, device, tch::Device::Cpu)
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    Env::new("CartPole-v0", obs_filter, act_filter, None).unwrap()
}

#[derive(Debug, Serialize)]
struct CartpoleRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f64>,
}

impl TryFrom<&Record> for CartpoleRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record
                .get_array1("obs")?
                .iter()
                .map(|v| *v as f64)
                .collect(),
        })
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("skip training")
                .long("skip_training")
                .takes_value(false)
                .help("Skip training"),
        )
        .get_matches();

    if !matches.is_present("skip training") {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent();
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(MODEL_DIR)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(MODEL_DIR);

        trainer.train(&mut recorder);
    }

    let mut env = create_env();
    let mut agent = create_agent();
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load(MODEL_DIR)?;
    agent.eval();

    util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("examples/model/iqn_cartpole_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?)?;
    }

    Ok(())
}
