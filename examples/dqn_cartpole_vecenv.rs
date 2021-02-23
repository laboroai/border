use std::{convert::TryFrom, fs::File, iter::FromIterator};
use serde::Serialize;
use anyhow::Result;
use csv::WriterBuilder;
use tch::nn;

use lrr::{
    core::{
        Trainer, Agent, util,
        record::{TensorboardRecorder, BufferedRecorder, Record}
    },
    env::py_gym_env::{
        Shape, PyVecGymEnv,
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_d::TchPyGymEnvDiscreteActBuffer,
        }
    },
    agent::{
        OptInterval,
        tch::{
            DQN, ReplayBuffer,
            model::Model1_1,
        }
    }
};

const N_PROCS: usize = 4;
const DIM_OBS: usize = 4;
const DIM_ACT: usize = 2;
const LR_QNET: f64 = 0.001;
const N_HIDDEN: i64 = 256;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.005;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(12);
const MAX_OPTS: usize = 1000;
const EVAL_INTERVAL: usize = 50;
const REPLAY_BUFFER_CAPACITY: usize = 2500;
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
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;

fn create_qnet() -> Model1_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim as _, N_HIDDEN, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", N_HIDDEN, out_dim as _, Default::default()));
    Model1_1::new(DIM_OBS, DIM_ACT, LR_QNET, network_fn)
}

fn create_agent() -> impl Agent<Env> {
    let qnet = create_qnet();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, N_PROCS);
    let agent: DQN<Env, _, _, _> = DQN::new(
        qnet,
        replay_buffer)
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU);
    agent
}

fn create_env(n_procs: usize) -> Env {
    let obs_filter = ObsFilter::vectorized();
    let act_filter = ActFilter::vectorized();
    Env::new("CartPole-v0", n_procs, obs_filter, act_filter, false).unwrap()
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);

    let env = create_env(N_PROCS);
    let env_eval = create_env(1);
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(MAX_OPTS)
        .eval_interval(EVAL_INTERVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL);
    let mut recorder = TensorboardRecorder::new("./examples/model/dqn_cartpole_vecenv");

    trainer.train(&mut recorder);
    trainer.get_agent().save("./examples/model/dqn_cartpole_vecenv").unwrap();

    trainer.get_env().close();
    trainer.get_env_eval().close();

    Ok(())
}
