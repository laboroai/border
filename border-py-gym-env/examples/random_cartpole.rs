use anyhow::Result;
use border_core::{
    record::{BufferedRecorder, Record},
    util, Env as _, Policy,
};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
    PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};
use serde::Serialize;
use std::{convert::TryFrom, fs::File};

type PyObsDtype = f32;

type Obs = PyGymEnvObs<PyObsDtype, f32>;
type Act = PyGymEnvDiscreteAct;
type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

#[derive(Clone)]
struct RandomPolicyConfig;

struct RandomPolicy;

impl Policy<Env> for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(_config: Self::Config) -> Self {
        Self
    }

    fn sample(&mut self, _: &Obs) -> Act {
        let v = fastrand::u32(..=1);
        Act::new(vec![v as i32])
    }
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
    fastrand::seed(42);

    let env_config = PyGymEnvConfig::default()
        .name("CartPole-v1".to_string())
        .render_mode(Some("human".to_string()))
        .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(
            "border-py-gym-env/examples/random_cartpole_eval.csv",
        )?);
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?)?;
    }

    Ok(())
}

#[test]
fn test_random_cartpole() {
    fastrand::seed(42);

    let env_config = PyGymEnvConfig::default()
        .name("CartPole-v1".to_string())
        .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
    let mut env = Env::build(&env_config, 0).unwrap();
    let mut recorder = BufferedRecorder::new();
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 1, &mut recorder).unwrap();
}
