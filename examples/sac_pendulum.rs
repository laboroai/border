use std::{convert::TryFrom, fs::File, iter::FromIterator};
use serde::Serialize;
use anyhow::Result;
use csv::WriterBuilder;
use tch::nn;
use pyo3::PyObject;

use lrr::{
    core::{
        Agent, Trainer,
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
            SAC, ReplayBuffer,
            model::{Model1_2, Model2_1},
        }
    }
};

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[3]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[1]
    }

    fn squeeze_first_dim() -> bool {
        true
    }
}

fn create_actor() -> Model1_2 {
    let network_fn = |p: &nn::Path, in_dim, hidden_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", 64, hidden_dim as _, Default::default()));
    Model1_2::new(3, 64, 1, 3e-4, network_fn)
}

fn create_critic() -> Model2_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl3", 64, out_dim as _, Default::default()));
        Model2_1::new(4, 1, 3e-4, network_fn)
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
    let actor = create_actor();
    let critic = create_critic();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(100_000, 1);
    let agent: SAC<Env, _, _, _, _> = SAC::new(
        critic,
        actor,
        replay_buffer)
        .opt_interval(OptInterval::Steps(1))
        .n_updates_per_opt(1)
        .min_transitions_warmup(1000)
        .batch_size(128)
        .discount_factor(0.99)
        .tau(0.001)
        .alpha(1.0);
    agent
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default(); //new();
    let act_filter = ActFilter {}; //new();
    Env::new("Pendulum-v0", obs_filter, act_filter)
        .unwrap()
        .max_steps(Some(200))
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
            //     .collect() //map(|v| *v as f64)
            // )
        })
    }
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);

    let env = create_env();
    let env_eval = create_env();
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(200_000)
        .eval_interval(10_000)
        .n_episodes_per_eval(5);
    let mut recorder = TensorboardRecorder::new("./examples/model/sac_pendulum");

    trainer.train(&mut recorder);
    trainer.get_agent().save("./examples/model/sac_pendulum").unwrap();

    let mut env = create_env();
    let mut agent = create_agent();
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load("./examples/model/sac_pendulum").unwrap();
    agent.eval();

    eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = csv::WriterBuilder::new().has_headers(false)
        .from_writer(File::create("examples/model/sac_pendulum_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(PendulumRecord::try_from(record)?)?;
    }

    Ok(())
}
