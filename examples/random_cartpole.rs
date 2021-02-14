use std::{io, convert::TryFrom};
use serde::Serialize;
use anyhow::Result;

use lrr::{
    error::LrrError,
    core::{Policy, util, record::{Record, BufferedRecorder}},
    env::py_gym_env::{
        PyGymEnv, 
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter}
    },
    agent::tch::Shape
};

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

#[derive(Debug, Serialize)]
struct CartpoleRecord {
    reward: f32,
    // step: usize,
    // episode: usize,
}

impl TryFrom<&Record> for CartpoleRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            reward: record.get_scalar("reward")?
        })
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f64>;
type Act = PyGymEnvDiscreteAct;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

struct RandomPolicy {}

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        let v = fastrand::u32(..=1);
        Act::new(vec![v as i32])
    }
}

// fn main() -> Result<(), Box<dyn Error>> {
fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let obs_filter = ObsFilter::default(); //new();
    let act_filter = ActFilter::default(); //new();
    let mut recorder = BufferedRecorder::new();
    // let mut env = Env::new("CartPole-v0", obs_filter, act_filter)?;
    let mut env = Env::new("CartPole-v0", obs_filter, act_filter).unwrap();
    env.set_render(true);
    let mut policy = RandomPolicy{};

    util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder);

    let mut wtr = csv::Writer::from_writer(io::stdout());
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?);
    }

    Ok(())
}
