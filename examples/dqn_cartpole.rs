use anyhow::Result;
use border_core::{
    record::{BufferedRecorder, Record, TensorboardRecorder},
    util, Agent, TrainerBuilder,
};
use border_py_gym_env::{
    PyGymEnv, Shape, shape, newtype_obs, newtype_act_d,
    // PyGymEnvObs, PyGymEnvObsRawFilter,
    // PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
    // tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},

};
use border_tch_agent::{
    dqn::{DQNBuilder, DQNExplorer, EpsilonGreedy},
    util::OptInterval,
};
use clap::{App, Arg};
use csv::WriterBuilder;
use serde::Serialize;
use std::{convert::TryFrom, fs::File};

const DIM_OBS: i64 = 4;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.005;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(50);
const MAX_OPTS: usize = 1000;
const EVAL_INTERVAL: usize = 50;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const MODEL_DIR: &str = "./examples/model/dqn_cartpole";

shape!(ObsShape, [DIM_OBS as usize]);
newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
newtype_act_d!(Act, ActFilter);

// type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64, f32>;
// type ActFilter = PyGymEnvDiscreteActRawFilter;
// type Obs = PyGymEnvObs<ObsShape, f64, f32>;
// type Act = PyGymEnvDiscreteAct;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64, f32>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;

mod dqn_model {
    use anyhow::Result;
    use border_tch_agent::{
        dqn::{DQNModel, DQNModelBuilder},
        model::SubModel,
        util::OutDim,
    };
    use serde::{Deserialize, Serialize};
    use tch::{nn, nn::Module, Device, Tensor};

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
    pub struct MLPConfig {
        in_dim: i64,
        out_dim: i64,
    }

    impl OutDim for MLPConfig {
        fn get_out_dim(&self) -> i64 {
            self.out_dim
        }

        fn set_out_dim(&mut self, v: i64) {
            self.out_dim = v;
        }
    }

    impl MLPConfig {
        fn new(in_dim: i64, out_dim: i64) -> Self {
            Self { in_dim, out_dim }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    // Two layer perceptron
    pub struct MLP {
        in_dim: i64,
        out_dim: i64,
        device: Device,
        seq: nn::Sequential,
    }

    impl MLP {
        fn create_net(var_store: &nn::VarStore, in_dim: i64, out_dim: i64) -> nn::Sequential {
            let p = &var_store.root();
            nn::seq()
                .add(nn::linear(p / "cl1", in_dim, 256, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "cl2", 256, out_dim, Default::default()))
        }
    }

    impl SubModel for MLP {
        type Config = MLPConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, x: &Self::Input) -> Tensor {
            self.seq.forward(&x.to(self.device))
        }

        fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
            let in_dim = config.in_dim;
            let out_dim = config.out_dim;
            let device = var_store.device();
            let seq = Self::create_net(var_store, in_dim, out_dim);

            Self {
                in_dim,
                out_dim,
                device,
                seq,
            }
        }

        fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
            let in_dim = self.in_dim;
            let out_dim = self.out_dim;
            let device = var_store.device();
            let seq = Self::create_net(&var_store, in_dim, out_dim);

            Self {
                in_dim,
                out_dim,
                device,
                seq,
            }
        }
    }

    // DQN model
    pub fn create_dqn_model(
        in_dim: i64,
        out_dim: i64,
        learning_rate: f64,
        device: Device,
    ) -> Result<DQNModel<MLP>> {
        let q_config = MLPConfig::new(in_dim, out_dim);
        DQNModelBuilder::default()
            .opt_config(border_tch_agent::opt::OptimizerConfig::Adam { lr: learning_rate })
            .build_with_submodel_configs(q_config, device)
    }
}

fn create_agent(epsilon_greedy: bool) -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let qnet = dqn_model::create_dqn_model(DIM_OBS, DIM_ACT, LR_CRITIC, device)?;
    let builder = DQNBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY);

    Ok(if epsilon_greedy {
        builder.explorer(DQNExplorer::EpsilonGreedy(EpsilonGreedy::new()))
    } else {
        builder
    }
    .build::<_, _, ObsBuffer, ActBuffer>(qnet, device))
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
            // obs: Vec::from_iter(
            //     record.get_array1("obs")?.iter().map(|v| *v as f64)
            // )
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
        .arg(
            Arg::with_name("egreedy")
                .long("epsilon_greedy")
                .takes_value(false)
                .help("Epsilon greedy"),
        )
        .get_matches();

    if !matches.is_present("skip training") {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent(matches.is_present("egreddy"))?;
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
    let mut agent = create_agent(matches.is_present("egreddy"))?;
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load(MODEL_DIR).unwrap(); // TODO: define appropriate error
    agent.eval();

    util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("examples/model/dqn_cartpole_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?)?;
    }

    Ok(())
}
