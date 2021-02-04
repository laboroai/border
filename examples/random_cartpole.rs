use std::error::Error;
use lrr::core::{Policy, util};
use lrr::py_gym_env::{
    PyGymEnv, PyGymEnvObs, PyGymEnvObsRawFilter,
    act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter}
};
use lrr::agents::tch::Shape;

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
type Act = PyGymEnvDiscreteAct<ActFilter>;
type Env = PyGymEnv<Obs, Act, ObsFilter>;

struct RandomPolicy {}

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        let v = fastrand::u32(..=1);
        Act::new(vec![v as i32])
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let obs_filter = ObsFilter::new();
    let mut env = Env::new("CartPole-v0", obs_filter, false)?;
    env.set_render(true);
    let mut policy = RandomPolicy{};
    util::eval(&env, &mut policy, 5, None);

    Ok(())
}
