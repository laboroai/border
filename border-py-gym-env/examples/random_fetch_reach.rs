use anyhow::Result;
use border_core::{
    Agent, Configurable, DefaultEvaluator, Evaluator as _, NullReplayBuffer, Policy,
};
use border_py_gym_env::{
    util::ArrayType, ArrayDictObsFilter, ArrayDictObsFilterConfig, ContinuousActFilter,
    GymActFilter, GymEnv, GymEnvConfig,
};
use ndarray::{Array, ArrayD};

mod obs {
    use border_py_gym_env::util::Array;

    #[derive(Clone, Debug)]
    pub struct Obs(Vec<(String, Array)>);

    impl border_core::Obs for Obs {
        fn len(&self) -> usize {
            match self.0.get(0) {
                None => 0,
                Some(v) => v.1.len(),
            }
        }
    }

    impl From<Vec<(String, Array)>> for Obs {
        fn from(obs: Vec<(String, Array)>) -> Self {
            Obs(obs)
        }
    }
}

mod act {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct Act(ArrayD<f32>);

    impl border_core::Act for Act {}

    impl Act {
        pub fn new(a: ArrayD<f32>) -> Self {
            Self(a)
        }
    }

    impl From<Act> for ArrayD<f32> {
        fn from(value: Act) -> Self {
            value.0
        }
    }
}

use act::Act;
use obs::Obs;
use serde::Deserialize;

type ObsFilter = ArrayDictObsFilter<Obs>;
type ActFilter = ContinuousActFilter<Act>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
type Evaluator = DefaultEvaluator<Env>;

#[derive(Clone, Deserialize)]
struct RandomPolicyConfig;

struct RandomPolicy;

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        let x = 2. * fastrand::f32() - 1.;
        // let y = 2. * fastrand::f32() - 1.;
        Act::new(Array::from(vec![x, x, x, x]).into_dyn())
    }
}

impl Configurable for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(_config: Self::Config) -> Self {
        Self
    }
}

impl Agent<Env, NullReplayBuffer> for RandomPolicy {}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let obs_filter_config = ArrayDictObsFilterConfig {
        key_and_types: vec![
            ("observation".into(), ArrayType::F32Array),
            ("desired_goal".into(), ArrayType::F32Array),
            ("achieved_goal".into(), ArrayType::F32Array),
        ],
        record_keys: None,
    };

    let env_config = GymEnvConfig::default()
        .name("FetchReach-v2".to_string())
        .render_mode(Some("human".to_string()))
        // .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .obs_filter_config(obs_filter_config)
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default());
    let mut policy = Box::new(RandomPolicy) as _;

    let _ = Evaluator::new(&env_config, 42, 5)?.evaluate(&mut policy);

    Ok(())
}
