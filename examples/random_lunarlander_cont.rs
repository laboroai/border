use anyhow::Result;
use border_core::{
    record::{BufferedRecorder},
    shape, util, Policy,
};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvBuilder, PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter, PyGymEnvObs,
    PyGymEnvObsRawFilter,
};
use ndarray::Array;

shape!(ObsShape, [8]);
shape!(ActShape, [2]);

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

struct RandomPolicy {}

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        Act::new(Array::from(vec![0.0, 0.0]).into_dyn())
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    let mut recorder = BufferedRecorder::new();
    let mut env = PyGymEnvBuilder::default()
        .build("LunarLanderContinuous-v2", obs_filter, act_filter)
        .unwrap();
    env.set_render(true);
    let mut policy = RandomPolicy {};

    util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder);

    Ok(())
}
