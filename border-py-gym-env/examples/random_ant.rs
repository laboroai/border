use anyhow::Result;
use border_core::{
    Agent, Configurable, DefaultEvaluator, Evaluator as _, NullReplayBuffer, Policy,
};
use border_py_gym_env::{
    ndarray::{NdarrayConverter, NdarrayConverterConfig},
    GymEnv, GymEnvConfig, GymEnvConverter,
};
use ndarray::Array;
use serde::Deserialize;
use std::default::Default;

type Env = GymEnv<NdarrayConverter>;
type Evaluator = DefaultEvaluator<Env>;
type Obs = <NdarrayConverter as GymEnvConverter>::Obs;
type Act = <NdarrayConverter as GymEnvConverter>::Act;

#[derive(Clone, Deserialize)]
struct RandomPolicyConfig;

struct RandomPolicy;

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        Act::new_cont(
            Array::from(
                (0..8)
                    .map(|_| 2f32 * fastrand::f32() - 1f32)
                    .collect::<Vec<_>>(),
            )
            .into_dyn(),
        )
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

    let env_config = GymEnvConfig::default()
        .name("Ant-v4".to_string())
        .converter_config(NdarrayConverterConfig {})
        .render_mode(Some("human".to_string()));
    let mut policy = Box::new(RandomPolicy) as _;

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut policy);

    Ok(())
}

#[test]
fn test_random_ant() -> Result<()> {
    fastrand::seed(42);

    let env_config = GymEnvConfig::default()
        .name("Ant-v4".to_string())
        .converter_config(NdarrayConverterConfig {});
    let mut policy = Box::new(RandomPolicy) as _;

    let _ = Evaluator::new(&env_config, 0, 1)?.evaluate(&mut policy);

    Ok(())
}
