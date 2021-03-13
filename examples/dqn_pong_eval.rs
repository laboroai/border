use anyhow::Result;
use tch::nn;

use border::{
    core::{
        Agent,
    },
    env::py_gym_env::{
        Shape, PyGymEnv,
        obs::PyGymEnvObs,
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_d::TchPyGymEnvDiscreteActBuffer
        }
    },
    agent::{
        tch::{
            DQNBuilder, ReplayBuffer as ReplayBuffer_, model::Model1_1,
        }
    }
};

const N_STACK: usize = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
const DIM_ACT: usize = 6;

const LR_QNET: f64 = 0.0;
const REPLAY_BUFFER_CAPACITY: usize = 1;

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
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
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
    let replay_buffer = ReplayBuffer::new(REPLAY_BUFFER_CAPACITY, 1);

    DQNBuilder::new()
        .build(qnet, replay_buffer, device)
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::new(N_STACK as i64);
    let act_filter = ActFilter::default();
    Env::new("PongNoFrameskip-v4", obs_filter, act_filter, true).unwrap()
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);

    let mut agent = create_agent();
    let mut env_eval = create_env();

    env_eval.set_render(true);
    env_eval.set_wait_in_render(std::time::Duration::from_millis(5));

    agent.load("./examples/model/dqn_pong_vecenv_20210307_ec2").unwrap(); // TODO: define appropriate error
    agent.eval();
    let reward = border::core::util::eval(&mut env_eval, &mut agent, 5);
    println!("{:?}", reward);
 
    Ok(())
}