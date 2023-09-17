use anyhow::Result;
use border_core::{record::BufferedRecorder, util, Env as _, Policy};
use border_py_gym_env::{
    GymEnv, GymActFilter, GymEnvConfig, GymContinuousAct,
    ContinuousActFilter, GymObsFilter, ArrayObsFilter,
};
use ndarray::{Array, ArrayD, IxDyn};
use std::default::Default;

mod obs {
    use super::*;
    
    #[derive(Clone, Debug)]
    pub struct Obs(ArrayD<f32>);

    impl border_core::Obs for Obs {
        fn dummy(_n: usize) -> Self {
            Self(ArrayD::zeros(IxDyn(&[0])))
        }

        fn len(&self) -> usize {
            self.0.shape()[0]
        }
    }

    impl From<ArrayD<f32>> for Obs {
        fn from(obs: ArrayD<f32>) -> Self {
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

use obs::Obs;
use act::Act;

type ObsFilter = ArrayObsFilter<f32, f32, Obs>;
type ActFilter = ContinuousActFilter<Act>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;

#[derive(Clone)]
struct RandomPolicyConfig;

struct RandomPolicy;

impl Policy<Env> for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(_config: Self::Config) -> Self {
        Self
    }

    fn sample(&mut self, _: &Obs) -> Act {
        Act::new(
            Array::from(
                (0..8)
                    .map(|_| 2f32 * fastrand::f32() - 1f32)
                    .collect::<Vec<_>>(),
            )
            .into_dyn(),
        )
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("Ant-v4".to_string())
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default())
        .render_mode(Some("human".to_string()));
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();
    let mut policy = RandomPolicy;
    // env.set_render(true);

    let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

    Ok(())
}

#[test]
fn test_random_ant() {
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("Ant-v4".to_string())
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default())
        .pybullet(true);
    let mut env = Env::build(&env_config, 0).unwrap();
    let mut recorder = BufferedRecorder::new();
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 1, &mut recorder).unwrap();
}
