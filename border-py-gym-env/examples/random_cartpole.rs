use anyhow::Result;
use border_core::{record::Record, DefaultEvaluator, Evaluator as _, Policy, Policy_};
use border_py_gym_env::{
    ArrayObsFilter, DiscreteActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use serde::Serialize;
use std::convert::TryFrom;

type PyObsDtype = f32;

mod obs {
    use ndarray::{ArrayD, IxDyn};

    #[derive(Clone, Debug)]
    pub struct CartPoleObs(ArrayD<f32>);

    impl border_core::Obs for CartPoleObs {
        fn len(&self) -> usize {
            self.0.shape()[0]
        }

        fn dummy(_n: usize) -> Self {
            Self(ArrayD::zeros(IxDyn(&[0])))
        }
    }

    impl From<ArrayD<f32>> for CartPoleObs {
        fn from(value: ArrayD<f32>) -> Self {
            Self(value)
        }
    }
}

mod act {
    #[derive(Clone, Debug)]
    pub struct CartPoleAct(Vec<i32>);

    impl CartPoleAct {
        pub fn new(v: Vec<i32>) -> Self {
            Self(v)
        }
    }

    impl border_core::Act for CartPoleAct {}

    impl From<CartPoleAct> for Vec<i32> {
        fn from(value: CartPoleAct) -> Self {
            value.0
        }
    }
}

use act::CartPoleAct;
use obs::CartPoleObs;

type Obs = CartPoleObs;
type Act = CartPoleAct;
type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
type ActFilter = DiscreteActFilter<Act>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
type Evaluator = DefaultEvaluator<Env, RandomPolicy>;

#[derive(Clone)]
struct RandomPolicyConfig;

struct RandomPolicy;

impl Policy_<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        let v = fastrand::u32(..=1);
        Act::new(vec![v as i32])
    }
}

impl Policy<Env> for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(_config: Self::Config) -> Self {
        Self
    }
}

#[derive(Debug, Serialize)]
struct CartpoleRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f64>,
}

impl TryFrom<&Record> for CartpoleRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record
                .get_array1("obs")?
                .iter()
                .map(|v| *v as f64)
                .collect(),
        })
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("CartPole-v1".to_string())
        .render_mode(Some("human".to_string()))
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default());
    let mut policy = RandomPolicy;

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut policy);

    // let mut wtr = csv::WriterBuilder::new()
    //     .has_headers(false)
    //     .from_writer(File::create(
    //         "border-py-gym-env/examples/random_cartpole_eval.csv",
    //     )?);
    // for record in recorder.iter() {
    //     wtr.serialize(CartpoleRecord::try_from(record)?)?;
    // }

    Ok(())
}

#[test]
fn test_random_cartpole() {
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("CartPole-v1".to_string())
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default());
    let mut policy = RandomPolicy;

    let _ = Evaluator::new(&env_config, 0, 5)
        .unwrap()
        .evaluate(&mut policy);
}
