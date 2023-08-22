use anyhow::Result;
use border_core::{DefaultEvaluator, Evaluator as _, Policy};
use border_py_gym_env::{
    GymEnv, GymActFilter, GymEnvConfig, GymContinuousAct,
    GymContinuousActRawFilter, GymObs, GymObsFilter, ArrayObsFilter,
};
use ndarray::Array;

type Obs = GymObs<f32, f32>;
type Act = GymContinuousAct;
type ObsFilter = ArrayObsFilter<f32, f32, Obs>;
type ActFilter = GymContinuousActRawFilter<Act>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
type Evaluator = DefaultEvaluator<Env, RandomPolicy>;

#[derive(Clone)]
struct RandomPolicyConfig;

struct RandomPolicy;

impl Policy<Env> for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(_config: Self::Config) -> Self {
        Self
    }

    fn sample(&mut self, _: &Obs) -> Act {
        let x = 2. * fastrand::f32() - 1.;
        let y = 2. * fastrand::f32() - 1.;
        Act::new(Array::from(vec![x, y]).into_dyn())
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("LunarLanderContinuous-v2".to_string())
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default())
        .render_mode(Some("human".to_string()));
    let mut policy = RandomPolicy;

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut policy);

    Ok(())
}

#[test]
fn test_lunalander_cont() {
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("LunarLanderContinuous-v2".to_string())
        .obs_filter_config(<ObsFilter as GymObsFilter<Obs>>::Config::default())
        .act_filter_config(<ActFilter as GymActFilter<Act>>::Config::default());
    let mut policy = RandomPolicy;

    let _ = Evaluator::new(&env_config, 0, 5).unwrap().evaluate(&mut policy);
}
