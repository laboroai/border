use std::{convert::TryFrom, fs::File, iter::FromIterator};
use serde::Serialize;
use anyhow::Result;
use tch::nn;
use pyo3::PyObject;

use border::{
    core::{
        Agent, TrainerBuilder,
        record::{BufferedRecorder, Record, RecordValue, TensorboardRecorder},
        util::eval_with_recorder
    }, 
    env::py_gym_env::{
        Shape, PyGymEnv, PyGymEnvActFilter,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_c::{PyGymEnvContinuousAct, to_pyobj},
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_c::TchPyGymEnvContinuousActBuffer
        }
    },
    agent::{
        OptInterval,
        tch::{
            SACBuilder, ReplayBuffer,
            model::{Model1_2, Model2_1},
        }
    }
};

const DIM_OBS: usize = 3;
const DIM_ACT: usize = 1;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 128;
const N_TRANSITIONS_WARMUP: usize = 1000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.001;
const ALPHA: f64 = 1.0;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 40_000;
const EVAL_INTERVAL: usize = 2_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const MAX_STEPS_IN_EPISODE: usize = 200;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[DIM_OBS]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[DIM_ACT]
    }

    fn squeeze_first_dim() -> bool {
        true
    }
}

fn create_actor(device: tch::Device) -> Model1_2 {
    let network_fn = |p: &nn::Path, in_dim, hidden_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", 64, hidden_dim as _, Default::default()));
    Model1_2::new(DIM_OBS, 64, DIM_ACT, LR_ACTOR, network_fn, device)
}

fn create_critic(device: tch::Device) -> Model2_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl3", 64, out_dim as _, Default::default()));
        Model2_1::new(DIM_OBS + DIM_ACT, 1, LR_CRITIC, network_fn, device)
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64>;
type Obs = PyGymEnvObs<ObsShape, f64>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

// Custom act filter
#[derive(Clone, Debug)]
struct ActFilter {}

impl PyGymEnvActFilter<Act> for ActFilter {
    fn filt(&mut self, act: Act) -> (PyObject, Record) {
        let act_filt = 2f32 * &act.act;
        let record = Record::from_slice(&[
            ("act_org", RecordValue::Array1(Vec::from_iter(act.act.iter().cloned()))),
            ("act_filt", RecordValue::Array1(Vec::from_iter(act_filt.iter().cloned())))
        ]);
        (to_pyobj::<ActShape>(act_filt), record)
    }
}

fn create_agent() -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(device);
    let critics = (0..1).map(|_| create_critic(device)).collect();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, 1);

    SACBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .alpha(ALPHA)
    .build(critics, actor, replay_buffer, device)
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter {};
    Env::new("Pendulum-v0", obs_filter, act_filter, false)
        .unwrap()
        .max_steps(Some(MAX_STEPS_IN_EPISODE))
}

#[derive(Debug, Serialize)]
struct PendulumRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f32>,
    act_org: Vec<f32>,
    act_filt: Vec<f32>,
}

impl TryFrom<&Record> for PendulumRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: Vec::from_iter(record.get_array1("obs")?.iter().cloned()),
            act_org: Vec::from_iter(record.get_array1("act_org")?.iter().cloned()),
            act_filt: Vec::from_iter(record.get_array1("act_filt")?.iter().cloned()),
        })
    }
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);

    let env = create_env();
    let env_eval = create_env();
    let agent = create_agent();
    let mut trainer = TrainerBuilder::default()
        .max_opts(MAX_OPTS)
        .eval_interval(EVAL_INTERVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL)
        .build(env, env_eval, agent);
    let mut recorder = TensorboardRecorder::new("./examples/model/sac_pendulum");

    trainer.train(&mut recorder);
    trainer.get_agent().save("./examples/model/sac_pendulum").unwrap();

    let mut env = create_env();
    let mut agent = create_agent();
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load("./examples/model/sac_pendulum").unwrap();
    agent.eval();

    let reward = eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);
    println!("{:?}", reward);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = csv::WriterBuilder::new().has_headers(false)
        .from_writer(File::create("examples/model/sac_pendulum_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(PendulumRecord::try_from(record)?)?;
    }

    Ok(())
}
