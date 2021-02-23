use std::cell::RefCell;
use anyhow::Result;

use lrr::{
    core::{
        Policy, util, Env as EnvTrait,
    },
    env::py_gym_env::{
        Shape, PyVecGymEnv,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter}
    },
};

const N_PROCS: usize = 4;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f64>;
type Act = PyGymEnvDiscreteAct;
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;

struct RandomPolicy {}

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        Act::new((0..N_PROCS).map(|_| fastrand::u32(..=1) as i32).collect())
    }
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::vectorized();
    let act_filter = ActFilter::vectorized();
    Env::new("CartPole-v0", N_PROCS, obs_filter, act_filter, false).unwrap()
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let mut policy = RandomPolicy{};
    let mut env = create_env();
    let obs = env.reset(None).unwrap();
    let obs_prev = RefCell::new(Some(obs));

    for _ in 0..200 {
        let _step = util::sample(&mut env, &mut policy, &obs_prev);
    }

    env.close();

    Ok(())
}
