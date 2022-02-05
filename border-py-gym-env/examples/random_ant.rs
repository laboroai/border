use anyhow::Result;
use border_core::{record::BufferedRecorder, shape, util, Env as _, Policy};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvContinuousAct,
    PyGymEnvContinuousActRawFilter, PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};
use ndarray::Array;
use std::default::Default;

shape!(ObsShape, [28]);
shape!(ActShape, [8]);

type Obs = PyGymEnvObs<ObsShape, f32, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32, f32, Obs>;
type ActFilter = PyGymEnvContinuousActRawFilter<ActShape, Act>;
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
        Act::new(
            Array::from(
                (0..8)
                    .map(|_| 2f32 * fastrand::f32() - 1f32)
                    .collect::<Vec<_>>(),
            )
            .into_dyn(),
        )
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = PyGymEnvConfig::default()
        .name("AntPyBulletEnv-v0".to_string())
        .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default())
        .pybullet(true);
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

    Ok(())
}

#[test]
fn test_random_ant() {
    fastrand::seed(42);

    let env_config = PyGymEnvConfig::default()
        .name("AntPyBulletEnv-v0".to_string())
        .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default())
        .pybullet(true);
    let mut env = Env::build(&env_config, 0).unwrap();
    let mut recorder = BufferedRecorder::new();
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 1, &mut recorder).unwrap();
}
