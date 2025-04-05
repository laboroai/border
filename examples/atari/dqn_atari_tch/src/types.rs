//! Type aliases
use border_atari_env::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_tch_agent::{cnn::AtariCnn, dqn::Dqn as Dqn_, TensorBatch};
use border_core::{
    generic_replay_buffer::{SimpleReplayBuffer, SimpleStepProcessor},
    DefaultEvaluator,
};

pub type Obs = BorderAtariObs;
pub type Act = BorderAtariAct;
pub type ObsBatch = TensorBatch;
pub type ActBatch = TensorBatch;
pub type ObsFilter = BorderAtariObsRawFilter<Obs>;
pub type ActFilter = BorderAtariActRawFilter<Act>;
pub type EnvConfig = BorderAtariEnvConfig<Obs, Act, ObsFilter, ActFilter>;
pub type Env = BorderAtariEnv<Obs, Act, ObsFilter, ActFilter>;
pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
pub type Dqn = Dqn_<Env, AtariCnn, ReplayBuffer>;
pub type Evaluator = DefaultEvaluator<Env>; 