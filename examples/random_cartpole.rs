use std::error::Error;
use lrr::core::{Policy, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::Shape;
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvDiscreteAct};
use lrr::agents::tch::py_gym_env::obs::TchPyGymEnvObsRawFilter;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

type O = TchPyGymEnvObs<ObsShape, f64>;
type A = TchPyGymEnvDiscreteAct;
type E = PyGymEnv<O, A, TchPyGymEnvObsRawFilter<ObsShape, f64>>;

struct RandomPolicy {}

impl Policy<E> for RandomPolicy {
    fn sample(&mut self, _: &O) -> TchPyGymEnvDiscreteAct {
        let v = fastrand::u32(..=1);
        TchPyGymEnvDiscreteAct(vec![v as i32])
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let obs_filter = TchPyGymEnvObsRawFilter::new();
    let mut env = E::new("CartPole-v0", obs_filter, false)?;
    env.set_render(true);
    let mut policy = RandomPolicy{};
    util::eval(&env, &mut policy, 5, None);

    Ok(())
}
