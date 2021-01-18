use std::error::Error;
use lrr::core::{Policy, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::Shape;
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvDiscreteAct};

#[derive(Debug, Clone)]
struct CartPoleObsShape {}

impl Shape for CartPoleObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

type O = TchPyGymEnvObs<CartPoleObsShape, f64>;
type A = TchPyGymEnvDiscreteAct;
type E = PyGymEnv<O, A>;

struct RandomPolicy {}

impl Policy<E> for RandomPolicy {
    fn sample(&mut self, _: &O) -> TchPyGymEnvDiscreteAct {
        let v = fastrand::u32(..=1);
        TchPyGymEnvDiscreteAct(vec![v as i32])
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let mut env = E::new("CartPole-v0", false)?;
    env.set_render(true);
    let mut policy = RandomPolicy{};
    util::eval(&env, &mut policy, 5, None);

    Ok(())
}
