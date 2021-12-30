//! Utilities for test.
use anyhow::Result;
use crate::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_core::{Agent as Agent_, Policy, ReplayBufferBase, replay_buffer::{SimpleReplayBuffer, SubBatch}};

pub type Obs = BorderAtariObs;
pub type Act = BorderAtariAct;
pub type ObsFilter = BorderAtariObsRawFilter<Obs>;
pub type ActFilter = BorderAtariActRawFilter<Act>;
pub type EnvConfig = BorderAtariEnvConfig<Obs, Act, ObsFilter, ActFilter>;
pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
pub type Env = BorderAtariEnv<Obs, Act, ObsFilter, ActFilter>;
pub type Agent = RandomAgent;

/// Consists the observation part of a batch in [SimpleReplayBuffer].
pub struct ObsBatch {
    pub obs: Vec<u8>,
}

impl SubBatch for ObsBatch {
    fn new(_capacity: usize) -> Self {
        unimplemented!();
    }

    fn push(&mut self, _i: usize, _data: &Self) {
        unimplemented!();
    }

    fn sample(&self, _ixs: &Vec<usize>) -> Self {
        unimplemented!();
    }
}

/// Consists the action part of a batch in [SimpleReplayBuffer].
pub struct ActBatch {
    pub obs: Vec<u8>,
}

impl SubBatch for ActBatch {
    fn new(_capacity: usize) -> Self {
        unimplemented!();
    }

    fn push(&mut self, _i: usize, _data: &Self) {
        unimplemented!();
    }

    fn sample(&self, _ixs: &Vec<usize>) -> Self {
        unimplemented!();
    }
}

#[derive(Clone)]
/// Configuration of [RandomAgent].
pub struct RandomAgentConfig {
    pub n_acts: usize,
}

/// A random policy.
pub struct RandomAgent {
    n_acts: usize,
}

impl Policy<Env> for RandomAgent {
    type Config = RandomAgentConfig;

    fn build(config: Self::Config) -> Self {
        Self {
            n_acts: config.n_acts,
        }
    }

    fn sample(&mut self, _: &Obs) -> Act {
        fastrand::u8(..self.n_acts as u8).into()
    }
}

impl<R: ReplayBufferBase> Agent_<Env, R> for RandomAgent
{
    fn train(&mut self) {
        unimplemented!();
    }

    fn eval(&mut self) {
        unimplemented!();
    }

    fn is_train(&self) -> bool {
        unimplemented!();
    }

    fn opt(&mut self, _buffer: &mut R) -> Option<border_core::record::Record> {
        unimplemented!();
    }

    fn save<T: AsRef<std::path::Path>>(&self, _path: T) -> Result<()> {
        unimplemented!();
    }

    fn load<T: AsRef<std::path::Path>>(&mut self, _path: T) -> Result<()> {
        unimplemented!();
    }
}

/// Returns the default configuration of [BorderAtariEnv].
pub fn env_config(name: String) -> EnvConfig {
    EnvConfig::default().name(name)
}

// fn main() -> Result<()> {
//     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
//     fastrand::seed(42);

//     let env_config = env_config("pong".to_string());
//     let mut env = Env::build(&env_config, 42)?;
//     let mut recorder = BufferedRecorder::new();
//     let n_acts = env.get_num_actions_atari();
//     let policy_config = RandomPolicyConfig {
//         n_acts: n_acts as _,
//     };
//     let mut policy = RandomPolicy::build(policy_config);

//     env.open()?;
//     let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

//     Ok(())
// }
