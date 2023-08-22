use anyhow::Result;
use border_core::{record::BufferedRecorder, util, Env as _, Policy};
use border_py_gym_env::{
    GymEnv, GymActFilter, GymEnvConfig, GymContinuousAct,
    GymContinuousActRawFilter, GymObs, GymObsFilter, ArrayObsFilter,
};
use ndarray::Array;
use std::default::Default;

type Obs = GymObs<f32, f32>;
type Act = GymContinuousAct;
type ObsFilter = ArrayObsFilter<f32, f32, Obs>;
type ActFilter = GymContinuousActRawFilter<Act>;
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
        .name("AntPyBulletEnv-v0".to_string())
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default())
        .pybullet(true);
    let mut env = Env::build(&env_config, 0).unwrap();
    let mut recorder = BufferedRecorder::new();
    let mut policy = RandomPolicy;

    let _ = util::eval_with_recorder(&mut env, &mut policy, 1, &mut recorder).unwrap();
}
