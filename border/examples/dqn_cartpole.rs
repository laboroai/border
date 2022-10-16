use anyhow::Result;
use border_core::{
    record::{BufferedRecorder, Record, TensorboardRecorder},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig, SubBatch,
    },
    shape, util, Agent, Env as _, Policy, Trainer, TrainerConfig,
};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
    PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};
use border_tch_agent::{
    dqn::{DqnConfig, DqnModelConfig, Dqn},
    mlp::{MlpConfig, Mlp},
    TensorSubBatch,
};
use clap::{App, Arg};
use csv::WriterBuilder;
use serde::Serialize;
use std::{convert::TryFrom, fs::File};
use tch::Tensor;

const DIM_OBS: i64 = 4;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.005;
const OPT_INTERVAL: usize = 50;
const MAX_OPTS: usize = 1000;
const EVAL_INTERVAL: usize = 50;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";

shape!(ObsShape, [DIM_OBS as usize]);
shape!(ActShape, [1]);

type PyObsDtype = f32;

#[derive(Clone, Debug)]
struct Obs(PyGymEnvObs<ObsShape, PyObsDtype, f32>);

impl border_core::Obs for Obs {
    fn dummy(n: usize) -> Self {
        Obs(PyGymEnvObs::dummy(n))
    }

    fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
        Obs(self.0.merge(obs_reset.0, is_done))
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl From<PyGymEnvObs<ObsShape, PyObsDtype, f32>> for Obs {
    fn from(obs: PyGymEnvObs<ObsShape, PyObsDtype, f32>) -> Self {
        Obs(obs)
    }
}

impl From<Obs> for Tensor {
    fn from(obs: Obs) -> Tensor {
        Tensor::try_from(&obs.0.obs).unwrap()
    }
}

struct ObsBatch(TensorSubBatch<ObsShape, f32>);

impl SubBatch for ObsBatch {
    fn new(capacity: usize) -> Self {
        Self(TensorSubBatch::new(capacity))
    }

    fn push(&mut self, i: usize, data: &Self) {
        self.0.push(i, &data.0)
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let buf = self.0.sample(ixs);
        Self(buf)
    }
}

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

impl From<ObsBatch> for Tensor {
    fn from(b: ObsBatch) -> Self {
        b.0.into()
    }
}

#[derive(Clone, Debug)]
struct Act(PyGymEnvDiscreteAct);

impl border_core::Act for Act {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl Into<PyGymEnvDiscreteAct> for Act {
    fn into(self) -> PyGymEnvDiscreteAct {
        self.0
    }
}

struct ActBatch(TensorSubBatch<ActShape, i64>);

impl SubBatch for ActBatch {
    fn new(capacity: usize) -> Self {
        Self(TensorSubBatch::new(capacity))
    }

    fn push(&mut self, i: usize, data: &Self) {
        self.0.push(i, &data.0)
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let buf = self.0.sample(ixs);
        Self(buf)
    }
}

impl From<Act> for Tensor {
    fn from(act: Act) -> Tensor {
        let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();

        // The first dimension of the action tensor is the number of processes,
        // which is 1 for the non-vectorized environment.
        t.unsqueeze(0)
    }
}

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

impl From<ActBatch> for Tensor {
    fn from(act: ActBatch) -> Self {
        act.0.into()
    }
}

impl From<Tensor> for Act {
    // `t` must be a 1-dimentional tensor of `f32`
    fn from(t: Tensor) -> Self {
        let data: Vec<i64> = t.into();
        let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
        Act(PyGymEnvDiscreteAct::new(data))
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, PyObsDtype, f32, Obs>;
type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;

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

fn create_agent(in_dim: i64, out_dim: i64) -> Dqn<Env, Mlp, ReplayBuffer> {
    let device = tch::Device::cuda_if_available();
    let config = {
        let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
        let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim);
        let model_config = DqnModelConfig::default()
            .q_config(mlp_config)
            .out_dim(out_dim)
            .opt_config(opt_config);
        DqnConfig::default()
            .n_updates_per_opt(N_UPDATES_PER_OPT)
            .min_transitions_warmup(N_TRANSITIONS_WARMUP)
            .batch_size(BATCH_SIZE)
            .discount_factor(DISCOUNT_FACTOR)
            .tau(TAU)
            .model_config(model_config)
            .device(device)
    };

    Dqn::build(config)
}

fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("CartPole-v0".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
}

fn train(max_opts: usize, model_dir: &str) -> Result<()> {
    let mut trainer = {
        let env_config = env_config();
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
        let config = TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(EVAL_INTERVAL)
            .record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .eval_episodes(N_EPISODES_PER_EVAL)
            .model_dir(model_dir);
        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            config,
            env_config,
            None,
            step_proc_config,
            replay_buffer_config,
        );

        trainer
    };

    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut agent = create_agent(DIM_OBS, DIM_ACT);

    trainer.train(&mut agent, &mut recorder)?;

    Ok(())
}

fn eval(model_dir: &str, render: bool) -> Result<()> {
    let mut env = Env::build(&env_config(), 0)?;
    let mut agent = create_agent(DIM_OBS, DIM_ACT);
    let mut recorder = BufferedRecorder::new();
    env.set_render(render);
    if render {
        env.set_wait_in_render(std::time::Duration::from_millis(10));
    }
    agent.load(model_dir)?;
    agent.eval();

    let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(model_dir.to_string() + "/eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(CartpoleRecord::try_from(record)?)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("skip training")
                .long("skip_training")
                .takes_value(false)
                .help("Skip training"),
        )
        .get_matches();

    if !matches.is_present("skip training") {
        train(MAX_OPTS, MODEL_DIR)?;
    }

    eval(&(MODEL_DIR.to_owned() + "/best"), true)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{train, eval};
    use anyhow::Result;
    use tempdir::TempDir;

    #[test]
    fn test_dqn_cartpole() -> Result<()> {
        let tmp_dir = TempDir::new("dqn_cartpole")?;
        let model_dir = match tmp_dir.as_ref().to_str() {
            Some(s) => s,
            None => panic!("Failed to get string of temporary directory"),
        };
        train(100, model_dir)?;
        eval(&(model_dir.to_owned() + "/best"), false)?;
        Ok(())
    }
}
