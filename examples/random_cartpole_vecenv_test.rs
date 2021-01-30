use std::error::Error;
use lrr::core::{Policy, util, Env as EnvTrait};
use lrr::py_gym_env::PyVecGymEnv;
use lrr::agents::tch::Shape;
use lrr::agents::tch::py_gym_env::obs::{TchPyGymEnvObs, TchPyGymEnvObsRawFilter};
use lrr::agents::tch::py_gym_env::act_d::{
    TchPyGymEnvDiscreteAct, TchPyGymEnvDiscreteActRawFilter
};

const N_PROCS: usize = 4;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

type ObsFilter = TchPyGymEnvObsRawFilter<ObsShape, f64>;
type ActFilter = TchPyGymEnvDiscreteActRawFilter;
type Obs = TchPyGymEnvObs<ObsShape, f64>;
type Act = TchPyGymEnvDiscreteAct<ActFilter>;
type Env = PyVecGymEnv<Obs, Act, ObsFilter>;

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

    let obs_filters: Vec<_> = (0..N_PROCS).map(|_| ObsFilter::new()).collect();
    let env = Env::new("CartPole-v0", obs_filters, false)?;
    let obs = env.reset(None);
    env.close();
    // env.set_render(true);
    // let mut policy = RandomPolicy{};
    // util::eval(&env, &mut policy, 5, None);

    Ok(())
}
