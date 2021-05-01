use anyhow::Result;
use clap::{App, Arg};
use std::time::Duration;
use tch::nn;

use border::{
    agent::{
        tch::{
            dqn::explorer::EpsilonGreedy, model::Model1_1, DQNBuilder,
            ReplayBuffer as ReplayBuffer_,
        },
        OptInterval,
    },
    core::{record::TensorboardRecorder, util, Agent, TrainerBuilder},
    env::py_gym_env::{
        AtariWrapper, PyGymEnv, PyGymEnvBuilder, Shape,
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        obs::PyGymEnvObs,
        framestack::FrameStackFilter,
        tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
    },
    util::url::get_model_from_url,
};

const N_PROCS: usize = 1;
const N_STACK: usize = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
const LR_QNET: f64 = 1e-4;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 32;
const N_TRANSITIONS_WARMUP: usize = 2500;
const N_UPDATES_PER_OPT: usize = 1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const SOFT_UPDATE_INTERVAL: usize = 10_000;
const TAU: f64 = 1.0;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 50_000;
const N_EPISODES_PER_EVAL: usize = 1;
const EPS_FINAL_STEP: usize = 1_000_000;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &DIM_OBS
    }
}

type ObsFilter = FrameStackFilter<ObsShape, u8, u8>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, u8, u8>;
type Act = PyGymEnvDiscreteAct;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8, u8>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;
type ReplayBuffer = ReplayBuffer_<Env, ObsBuffer, ActBuffer>;

fn stride(s: i64) -> nn::ConvConfig {
    nn::ConvConfig {
        stride: s,
        ..Default::default()
    }
}

fn create_critic(dim_act: usize, device: tch::Device) -> Model1_1 {
    let network_fn = |p: &nn::Path, _in_shape: &[usize], out_dim| {
        nn::seq()
            .add_fn(|xs| xs.squeeze1(2).internal_cast_float(true) / 255)
            .add(nn::conv2d(p / "c1", N_STACK as i64, 32, 8, stride(4)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
            .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 3136, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "l2", 512, out_dim as _, Default::default()))
    };
    Model1_1::new(&DIM_OBS, dim_act, LR_QNET, network_fn, device)
}

fn create_agent(dim_act: usize) -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let qnet = create_critic(dim_act, device);
    let replay_buffer = ReplayBuffer::new(REPLAY_BUFFER_CAPACITY, N_PROCS);

    DQNBuilder::new()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .soft_update_interval(SOFT_UPDATE_INTERVAL)
        .tau(TAU)
        .explorer(EpsilonGreedy::with_final_step(EPS_FINAL_STEP))
        .build(qnet, replay_buffer, device)
}

fn create_env(name: &str, mode: AtariWrapper) -> Env {
    let obs_filter = ObsFilter::new(N_STACK as i64);
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .atari_wrapper(Some(mode))
        .build(name, obs_filter, act_filter)
        .unwrap()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_atari")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("name")
                .long("name")
                .takes_value(true)
                .required(true)
                .index(1)
                .help("The name of the atari environment (e.g., PongNoFrameskip-v4)"),
        )
        .arg(
            Arg::with_name("play")
                .long("play")
                .takes_value(true)
                .help("Play with the trained model of the given path"),
        )
        .arg(
            Arg::with_name("play-gdrive")
                .long("play-gdrive")
                .takes_value(false)
                .help("Play with the trained model downloaded from google drive"),
        )
        .arg(
            Arg::with_name("wait")
                .long("wait")
                .takes_value(true)
                .default_value("25")
                .help("Waiting time in milliseconds between frames when playing"),
        )
        .get_matches();

    let name = matches.value_of("name").unwrap();
    let mut env_eval = create_env(name, AtariWrapper::Eval);
    let dim_act = env_eval.get_num_actions_atari();
    let mut agent = create_agent(dim_act as _);

    if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
        let env_train = create_env(name, AtariWrapper::Train);
        let saving_model_dir = format!("./examples/model/dqn_{}", name);
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(saving_model_dir)
            .build(env_train, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(format!("./examples/model/dqn_{}", name));
        trainer.train(&mut recorder);
    } else {
        if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        } else {
            // TODO: change file_base and url depending on the game
            let file_base = "dqn_PongNoFrameskip-v4_20210428_ec2";
            let url =
                "https://drive.google.com/uc?export=download&id=1TF5aN9fH5wd4APFHj9RP1JxuVNoi6lqJ";
            let model_dir = get_model_from_url(url, file_base)?;
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        };

        let time = matches.value_of("wait").unwrap().parse::<u64>()?;
        env_eval.set_render(true);
        env_eval.set_wait_in_render(Duration::from_millis(time));
        agent.eval();
        util::eval(&mut env_eval, &mut agent, 5);
    }

    Ok(())
}
