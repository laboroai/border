use std::{convert::TryFrom, fs::File, default::Default};
use serde::Serialize;
use anyhow::Result;
use clap::{Arg, App};
use csv::WriterBuilder;

use border::{
    core::{
        Agent, TrainerBuilder, util,
        record::{TensorboardRecorder, BufferedRecorder, Record}
    },
    env::py_gym_env::{
        Shape, PyGymEnv,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_d::TchPyGymEnvDiscreteActBuffer,
        }
    },
    agent::{
        OptInterval,
        tch::{
            ReplayBuffer,
            iqn::{IQNBuilder, EpsilonGreedy}
        }
    }
};

const DIM_OBS: i64 = 4;
const DIM_FEATURE: i64 = 32;
const DIM_EMBED: i64 = 64;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.001;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 10000;
const EVAL_INTERVAL: usize = 500;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const EPS_START: f64 = 1.0;
const EPS_FINAL: f64 = 0.01;
const FINAL_STEP: usize = 3500; // MAX_OPTS;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[DIM_OBS as _]
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64, f32>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f64, f32>;
type Act = PyGymEnvDiscreteAct;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64, f32>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;

mod iqn_model {
    use tch::{Tensor, Device, nn, nn::Module};
    use border::agent::tch::{
        model::SubModel,
        iqn::{IQNModel, IQNModelBuilder}
    };

    #[allow(clippy::upper_case_acronyms)]
    pub struct FCConfig {
        in_dim: i64,
        out_dim: i64,
        relu: bool,
    }

    impl FCConfig {
        fn new(in_dim: i64, out_dim: i64, relu: bool) -> Self {
            Self {
                in_dim,
                out_dim,
                relu
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
        seq: nn::Sequential
    }

    impl FC {
        fn create_net(var_store: &nn::VarStore, in_dim: i64, out_dim: i64, relu: bool) -> nn::Sequential {
            let p = &var_store.root();
            let mut seq = nn::seq()
                .add(nn::linear(p / "cl1", in_dim, out_dim, Default::default()));
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
                seq
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
                seq
            }
        }
    }

    // IQN model
    pub fn create_iqn_model(in_dim: i64, feature_dim: i64, embed_dim: i64,  out_dim: i64, learning_rate: f64,
        device: Device) -> IQNModel<FC, FC> {
        let fe_config = FCConfig::new(in_dim, feature_dim, true);
        let m_config = FCConfig::new(feature_dim, out_dim, false);
        IQNModelBuilder::default()
            .feature_dim(feature_dim)
            .embed_dim(embed_dim)
            .out_dim(out_dim)
            .learning_rate(learning_rate)
            .build(fe_config, m_config, device)
    }
}

fn create_agent() -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let iqn_model = iqn_model::create_iqn_model(DIM_OBS, DIM_FEATURE, DIM_EMBED, DIM_ACT, LR_CRITIC, device);
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, 1);
    IQNBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .explorer(EpsilonGreedy::with_params(EPS_START, EPS_FINAL, FINAL_STEP))
        .build(iqn_model, replay_buffer, device)
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default(); //::new();
    let act_filter = ActFilter::default();
    Env::new("CartPole-v0", obs_filter, act_filter, false).unwrap()
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
            obs: record.get_array1("obs")?.iter().map(|v| *v as f64).collect()
        })
    }
}

fn main() -> Result<()> {
    let matches = App::new("dqn_cartpole")
    .version("0.1.0")
    .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
    .arg(Arg::with_name("skip training")
        .long("skip_training")
        .takes_value(false)
        .help("Skip training"))
    .get_matches();

    env_logger::init();
    tch::manual_seed(42);

    if !matches.is_present("skip training") {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent();
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new("./examples/model/iqn_cartpole");
    
        trainer.train(&mut recorder);
        trainer.get_agent().save("./examples/model/iqn_cartpole").unwrap(); // TODO: define appropriate error    
    }

    let mut env = create_env();
    let mut agent = create_agent();
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load("./examples/model/iqn_cartpole").unwrap(); // TODO: define appropriate error
    agent.eval();

    util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = WriterBuilder::new().has_headers(false)
        .from_writer(File::create("examples/model/iqn_cartpole_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?)?;
    }

    Ok(())
}