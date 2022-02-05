use anyhow::Result;
use border_core::{
    Env as _,
    record::{BufferedRecorder},
    shape, util, Policy,
};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvConfig, PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter, PyGymEnvObs,
    PyGymEnvObsRawFilter, PyGymEnvObsFilter, PyGymEnvActFilter,
};
use ndarray::Array;

shape!(ObsShape, [8]);
shape!(ActShape, [2]);

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
        let x = 2. * fastrand::f32() - 1.;
        let y = 2. * fastrand::f32() - 1.;
        Act::new(Array::from(vec![x, y]).into_dyn())
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = PyGymEnvConfig::default()
        .name("LunarLanderContinuous-v2".to_string())
        .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

    Ok(())
}

#[test]
fn test_lunalander_cont() {
    fastrand::seed(42);

    let env_config = PyGymEnvConfig::default()
        .name("LunarLanderContinuous-v2".to_string())
        .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
    let mut env = Env::build(&env_config, 0).unwrap();
    let mut recorder = BufferedRecorder::new();
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 1, &mut recorder).unwrap();
}
