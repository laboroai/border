use anyhow::Result;
use tch::nn;

use border::{
    core::{
        TrainerBuilder, Agent,
        record::TensorboardRecorder,
    },
    env::py_gym_env::{
        Shape, PyVecGymEnv,
        obs::PyGymEnvObs,
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_d::TchPyGymEnvDiscreteActBuffer
        }
    },
    agent::{
        OptInterval,
        tch::{
            DQNBuilder, ReplayBuffer as ReplayBuffer_, model::Model1_1,
            dqn::explorer::{DQNExplorer, EpsilonGreedy},
        }
    }
};

const N_PROCS: usize = 4;
const N_STACK: usize = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
const DIM_ACT: usize = 6;

const LR_QNET: f64 = 1e-4;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 32;
const N_TRANSITIONS_WARMUP: usize = 2500;
const N_UPDATES_PER_OPT: usize = 1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const SOFT_UPDATE_INTERVAL: usize = 1_000;
const TAU: f64 = 1.0;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 10_000;
const N_EPISODES_PER_EVAL: usize = 1;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &DIM_OBS
    }
}

type ObsFilter = FrameStackFilter<ObsShape, u8>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, u8>;
type Act = PyGymEnvDiscreteAct;
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;
type ReplayBuffer = ReplayBuffer_<Env, ObsBuffer, ActBuffer>;

fn stride(s: i64) -> nn::ConvConfig {
    nn::ConvConfig {
        stride: s,
        ..Default::default()
    }
}

fn create_critic(device: tch::Device) -> Model1_1 {
    let network_fn = |p: &nn::Path, _in_shape: &[usize], out_dim| nn::seq()
        .add_fn(|xs| xs.squeeze1(2))
        .add(nn::conv2d(p / "c1", N_STACK as i64, 32, 8, stride(4)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
        .add_fn(|xs| xs.relu().flat_view())
        .add(nn::linear(p / "l1", 3136, 512, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "l2", 512, out_dim as _, Default::default()));
    Model1_1::new(&DIM_OBS, DIM_ACT, LR_QNET, network_fn, device)
}

fn create_agent() -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let qnet = create_critic(device);
    let replay_buffer = ReplayBuffer::new(REPLAY_BUFFER_CAPACITY, N_PROCS);

    DQNBuilder::new()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .soft_update_interval(SOFT_UPDATE_INTERVAL)
        .tau(TAU)
        .explorer(DQNExplorer::EpsilonGreedy(EpsilonGreedy::new()))
        .build(qnet, replay_buffer, device)
}

fn create_env(n_procs: usize) -> Env {
    let obs_filter = ObsFilter::vectorized(N_PROCS as i64, N_STACK as i64);
    let act_filter = ActFilter::vectorized();
    Env::new("PongNoFrameskip-v4", n_procs, obs_filter, act_filter, true).unwrap()
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);

    let mut agent = create_agent();
    let env = create_env(N_PROCS);
    let mut env_eval = create_env(1);

    let mut trainer = TrainerBuilder::default()
        .max_opts(MAX_OPTS)
        .eval_interval(EVAL_INTERVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL)
        .build(env, env_eval, agent);
    let mut recorder = TensorboardRecorder::new("./examples/model/dqn_pong_vecenv");

    trainer.train(&mut recorder);
    trainer.get_agent().save("./examples/model/dqn_pong_vecenv").unwrap(); // TODO: define appropriate error

    trainer.get_env().close();
    trainer.get_env_eval().close();

    Ok(())
}
