use std::time::Duration;
// use anyhow::Result;
use border::env::py_gym_env::{
    act_c::{PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter},
    obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
    PyGymEnv, PyGymEnvBuilder, Shape,
};
use border_core::{util, Policy};
use ndarray::Array;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[28]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[8]
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

struct RandomPolicy {}

impl Policy<Env> for RandomPolicy {
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

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    // TODO: Define appropriate error for failing to construct environment
    let mut env = PyGymEnvBuilder::default()
        .pybullet(true)
        .atari_wrapper(None)
        .build("AntPyBulletEnv-v0", obs_filter, act_filter)
        .unwrap();
    env.set_render(true);
    env.set_wait_in_render(Duration::from_millis(10));
    let mut policy = RandomPolicy {};

    util::eval(&mut env, &mut policy, 5);

    // Ok(())
}
