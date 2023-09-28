use anyhow::Result;
use border_core::{
    record::Record,
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig, SubBatch,
    },
    Agent, DefaultEvaluator, Evaluator as _, Policy, Trainer, TrainerConfig,
};
use border_py_gym_env::{
    util::vec_to_tensor, ArrayObsFilter, DiscreteActFilter, GymActFilter, GymEnv, GymEnvConfig,
    GymObsFilter,
};
use border_tch_agent::{
    iqn::{EpsilonGreedy, Iqn as Iqn_, IqnConfig, IqnModelConfig},
    mlp::{Mlp, MlpConfig},
    TensorSubBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::{App, Arg};
// use csv::WriterBuilder;
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;
use std::convert::TryFrom;
use tch::Tensor;

const DIM_OBS: i64 = 4;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DIM_FEATURE: i64 = 256;
const DIM_EMBED: i64 = 64;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.1; //0.005;
const SOFT_UPDATE_INTERVAL: usize = 100;
const OPT_INTERVAL: usize = 50;
const MAX_OPTS: usize = 10000;
const EVAL_INTERVAL: usize = 500;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const EPS_START: f64 = 1.0;
const EPS_FINAL: f64 = 0.1;
const FINAL_STEP: usize = MAX_OPTS;
const MODEL_DIR: &str = "border/examples/model/iqn_cartpole";

type PyObsDtype = f32;

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

    impl From<Obs> for Tensor {
        fn from(obs: Obs) -> Tensor {
            Tensor::try_from(&obs.0).unwrap()
        }
    }

    pub struct ObsBatch(TensorSubBatch);

    impl SubBatch for ObsBatch {
        fn new(capacity: usize) -> Self {
            Self(TensorSubBatch::new(capacity))
        }

        fn push(&mut self, i: usize, data: Self) {
            self.0.push(i, data.0)
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
}

mod act {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct Act(Vec<i32>);

    impl border_core::Act for Act {}

    impl From<Act> for Vec<i32> {
        fn from(value: Act) -> Self {
            value.0
        }
    }

    impl From<Tensor> for Act {
        // `t` must be a 1-dimentional tensor of `f32`
        fn from(t: Tensor) -> Self {
            let data: Vec<i64> = t.into();
            let data = data.iter().map(|&e| e as i32).collect();
            Act(data)
        }
    }

    pub struct ActBatch(TensorSubBatch);

    impl SubBatch for ActBatch {
        fn new(capacity: usize) -> Self {
            Self(TensorSubBatch::new(capacity))
        }

        fn push(&mut self, i: usize, data: Self) {
            self.0.push(i, data.0)
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let buf = self.0.sample(ixs);
            Self(buf)
        }
    }

    impl From<Act> for ActBatch {
        fn from(act: Act) -> Self {
            let t = vec_to_tensor::<_, i64>(act.0, true);
            Self(TensorSubBatch::from_tensor(t))
        }
    }

    // Required by Dqn
    impl From<ActBatch> for Tensor {
        fn from(act: ActBatch) -> Self {
            act.0.into()
        }
    }
}

use act::{Act, ActBatch};
use obs::{Obs, ObsBatch};

type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
type ActFilter = DiscreteActFilter<Act>;
type EnvConfig = GymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Iqn = Iqn_<Env, Mlp, Mlp, ReplayBuffer>;
type Evaluator = DefaultEvaluator<Env, Iqn>;

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

fn env_config() -> EnvConfig {
    GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("CartPole-v0".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
}

fn create_evaluator(env_config: &EnvConfig) -> Result<Evaluator> {
    Evaluator::new(env_config, 0, N_EPISODES_PER_EVAL)
}

fn create_agent(in_dim: i64, out_dim: i64) -> Iqn {
    let device = tch::Device::cuda_if_available();
    let config = {
        let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
        let f_config = MlpConfig::new(in_dim, vec![], DIM_FEATURE, true);
        let m_config = MlpConfig::new(DIM_FEATURE, vec![], out_dim, false);
        let model_config = IqnModelConfig::default()
            .feature_dim(DIM_FEATURE)
            .embed_dim(DIM_EMBED)
            .opt_config(opt_config)
            .f_config(f_config)
            .m_config(m_config);

        IqnConfig::default()
            .n_updates_per_opt(N_UPDATES_PER_OPT)
            .min_transitions_warmup(N_TRANSITIONS_WARMUP)
            .batch_size(BATCH_SIZE)
            .discount_factor(DISCOUNT_FACTOR)
            .tau(TAU)
            .model_config(model_config)
            .explorer(EpsilonGreedy::with_params(EPS_START, EPS_FINAL, FINAL_STEP))
            .soft_update_interval(SOFT_UPDATE_INTERVAL)
            .device(device)
    };

    Iqn::build(config)
}

fn train(max_opts: usize, model_dir: &str, eval_interval: usize) -> Result<()> {
    let mut trainer = {
        let env_config = env_config();
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
        let config = TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(eval_interval)
            .record_interval(eval_interval)
            .save_interval(eval_interval)
            .model_dir(model_dir);
        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            config,
            env_config,
            step_proc_config,
            replay_buffer_config,
        );

        trainer
    };
    let mut agent = create_agent(DIM_OBS, DIM_ACT);
    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut evaluator = create_evaluator(&env_config())?;

    trainer.train(&mut agent, &mut recorder, &mut evaluator)?;

    Ok(())
}

fn eval(model_dir: &str, render: bool) -> Result<()> {
    let env_config = {
        let mut env_config = env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        }
        env_config
    };
    let mut agent = {
        let mut agent = create_agent(DIM_OBS, DIM_ACT);
        agent.load(model_dir)?;
        agent.eval();
        agent
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <yoshioka@laboro.ai>")
        .arg(
            Arg::with_name("skip training")
                .long("skip-training")
                .takes_value(false)
                .help("Skip training"),
        )
        .get_matches();

    if !matches.is_present("skip training") {
        train(MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
    }

    eval(&(MODEL_DIR.to_owned() + "/best"), true)?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_iqn_cartpole() -> Result<()> {
        tch::manual_seed(42);

        let model_dir = TempDir::new("sac_pendulum")?;
        let model_dir = model_dir.path().to_str().unwrap();
        train(100, model_dir, 100)?;
        eval((model_dir.to_string() + "/best").as_str(), false)?;

        Ok(())
    }
}
