use std::error::Error;
use lrr::core::{Policy, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvContinuousAct, Shape};

#[derive(Debug, Clone)]
struct MountainCarObsShape {}

impl Shape for MountainCarObsShape {
    fn shape() -> &'static [usize] {
        &[2]
    }
}

type O = TchPyGymEnvObs<MountainCarObsShape>;
type A = TchPyGymEnvContinuousAct;
type E = PyGymEnv<O, A>;

struct RandomPolicy {}

impl Policy<E> for RandomPolicy {
    fn sample(&self, _: &O) -> A {
        A::new(vec![2.0 * fastrand::f32() - 1.0])
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // std::env::set_var("RUST_LOG", "trace");
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let mut env = E::new("MountainCarContinuous-v0", true)?;
    env.set_render(true);
    let policy = RandomPolicy{};
    util::eval(&env, &policy, 5, None);

    Ok(())
}
