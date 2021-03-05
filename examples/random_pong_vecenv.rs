use std::{convert::TryFrom, fs::File, iter::FromIterator, cell::RefCell};
use serde::Serialize;
use anyhow::Result;
use csv::WriterBuilder;
use tch::nn;

use lrr::{
    core::{Env as _, Policy, util, record::{BufferedRecorder, Record}},
    env::py_gym_env::{
        Shape, PyVecGymEnv,
        obs::PyGymEnvObs,
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
    },
};

const N_PROCS: usize = 4;
const N_STACK: usize = 4;
const DIM_OBS: (usize, usize, usize, usize) = (4, 1, 84, 84);
const DIM_ACT: usize = 6;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[DIM_OBS.0, DIM_OBS.1, DIM_OBS.2, DIM_OBS.3]
    }
}

type ObsFilter = FrameStackFilter<ObsShape, u8>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, u8>;
type Act = PyGymEnvDiscreteAct;
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;

fn create_policy() -> impl Policy<Env> {
    struct RandomPolicy {}

    impl Policy<Env> for RandomPolicy {
        fn sample(&mut self, _: &Obs) -> Act {
            Act::new((0..N_PROCS).map(|_| fastrand::u32(..DIM_ACT as u32) as i32).collect())
        }
    }
    
    RandomPolicy {}
}

fn create_env(n_procs: usize) -> Env {
    let obs_filter = ObsFilter::vectorized(N_PROCS as i64, N_STACK as i64);
    let act_filter = ActFilter::vectorized();
    Env::new("PongNoFrameskip-v4", n_procs, obs_filter, act_filter, true).unwrap()
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);

    let mut policy = create_policy();
    let mut env = create_env(N_PROCS);
    let obs = env.reset(None).unwrap();
    let obs_prev = RefCell::new(Some(obs));

    // for _ in 0..200 {
    //     let _step = util::sample(&mut env, &mut policy, &obs_prev);
    // }

    env.close();

    let mut env = create_env(1);
    util::eval(&mut env, &mut policy, 5);
    env.close();

    Ok(())
}
