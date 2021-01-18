use std::error::Error;
use lrr::core::{Policy, util};
use ndarray::Array;
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::Shape;
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvContinuousAct};

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

type O = TchPyGymEnvObs<ObsShape, f32>;
// type O = TchPyGymEnvObs<ObsShape, f64>; // Results in a runtime error in conversion of
// numpy array in lunarlander-cont-v2 environemnt beecause of type mismatch.
type A = TchPyGymEnvContinuousAct<ActShape>;
type E = PyGymEnv<O, A>;

struct RandomPolicy {}

impl Policy<E> for RandomPolicy {
    fn sample(&mut self, _: &O) -> A {
        // A::new(Array::from(vec![2.0 * fastrand::f32() - 1.0, 2.0 * fastrand::f32() - 1.0]).into_dyn())
        A::new(Array::from(vec![0.0, 0.0]).into_dyn())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let mut env = E::new("LunarLanderContinuous-v2", true)?;
    env.set_render(true);
    let mut policy = RandomPolicy{};
    util::eval(&env, &mut policy, 5, None);

    Ok(())
}
