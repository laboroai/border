use std::error::Error;
use ndarray::Array;
use lrr::{
    core::{Policy, util},
    py_gym_env::{
        PyGymEnv,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_c::{PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter}
    },
    agents::tch::Shape,
};

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[8]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[2]
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter; //<ActShape>;
type Obs = PyGymEnvObs<ObsShape, f32>;
// type O = TchPyGymEnvObs<ObsShape, f64>; // Results in a runtime error in conversion of
// numpy array in lunarlander-cont-v2 environemnt beecause of type mismatch.
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

struct RandomPolicy {}

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        // A::new(Array::from(vec![2.0 * fastrand::f32() - 1.0, 2.0 * fastrand::f32() - 1.0]).into_dyn())
        Act::new(Array::from(vec![0.0, 0.0]).into_dyn())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let obs_filter = ObsFilter::new();
    let act_filter = ActFilter::new();
    let mut env = Env::new("LunarLanderContinuous-v2", obs_filter, act_filter)?;
    env.set_render(true);
    let mut policy = RandomPolicy{};
    util::eval(&mut env, &mut policy, 5, None);

    Ok(())
}
