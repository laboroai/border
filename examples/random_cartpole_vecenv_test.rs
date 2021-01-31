use std::error::Error;
use std::cell::RefCell;
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
        Act::new((0..N_PROCS).map(|_| fastrand::u32(..=1) as i32).collect())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let obs_filters: Vec<_> = (0..N_PROCS).map(|_| ObsFilter::new()).collect();
    let env = Env::new("CartPole-v0", obs_filters, false)?;
    let mut policy = RandomPolicy{};

    let obs = env.reset(None).unwrap();
    let obs_prev = RefCell::new(Some(obs));

    for i in 0..200 {
        let step = util::sample(&env, &mut policy, &obs_prev);
    }

    env.close();
    // env.set_render(true);
    // util::eval(&env, &mut policy, 5, None);

    Ok(())
}
