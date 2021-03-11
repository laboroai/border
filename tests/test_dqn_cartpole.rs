use std::{convert::TryFrom, fs::File, iter::FromIterator};
use serde::Serialize;
use anyhow::Result;
use clap::{Arg, App};
use csv::WriterBuilder;
use tch::nn;

use border::{
    core::{
        Agent, Trainer, TrainerBuilder, util,
        record::{TensorboardRecorder, BufferedRecorder, Record}
    },
    env::py_gym_env::{
        Shape, PyGymEnv,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_d::TchPyGymEnvDiscreteActBuffer
        }
    },
    agent::{
        OptInterval,
        tch::{
            DQNBuilder, ReplayBuffer, model::Model1_1,
            dqn::explorer::{DQNExplorer, EpsilonGreedy},
        }
    }
};

const DIM_OBS: usize = 4;
const DIM_ACT: usize = 2;
const LR_QNET: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.005;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(50);
const MAX_OPTS: usize = 100;
const EVAL_INTERVAL: usize = 50;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[DIM_OBS]
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f64>;
type Act = PyGymEnvDiscreteAct;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;

fn create_critic(device: tch::Device) -> Model1_1 {
    let network_fn = |p: &nn::Path, in_dim: &[usize], out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim[0] as _, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", 256, out_dim as _, Default::default()));
    Model1_1::new(&[DIM_OBS], DIM_ACT, LR_QNET, network_fn, device)
}

fn create_agent(epsilon_greedy: bool) -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let qnet = create_critic(device);
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, 1);
    let builder = DQNBuilder::new()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU);

    if epsilon_greedy {
        builder.explorer(DQNExplorer::EpsilonGreedy(EpsilonGreedy::new()))
    }
    else {
        builder
    }
    .build(qnet, replay_buffer, device)
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
            obs: Vec::from_iter(
                record.get_array1("obs")?.iter().map(|v| *v as f64)
            )
        })
    }
}

#[test]
fn main() -> Result<()> {
    let matches = App::new("dqn_cartpole")
    .version("0.1.0")
    .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
    .arg(Arg::with_name("skip training")
        .long("skip_training")
        .takes_value(false)
        .help("Skip training"))
    .arg(Arg::with_name("egreedy")
        .long("epsilon_greedy")
        .takes_value(false)
        .help("Epsilon greedy"))
    .get_matches();

    env_logger::init();
    tch::manual_seed(42);

    if !matches.is_present("skip training") {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent(matches.is_present("egreddy"));
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new("./examples/model/dqn_cartpole");
    
        trainer.train(&mut recorder);
        trainer.get_agent().save("./examples/model/dqn_cartpole").unwrap(); // TODO: define appropriate error    
    }

    let mut env = create_env();
    let mut agent = create_agent(matches.is_present("egreddy"));
    let mut recorder = BufferedRecorder::new();
    // env.set_render(true);
    agent.load("./examples/model/dqn_cartpole").unwrap(); // TODO: define appropriate error
    agent.eval();

    util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = WriterBuilder::new().has_headers(false)
        .from_writer(File::create("examples/model/dqn_cartpole_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?)?;
    }

    Ok(())
}