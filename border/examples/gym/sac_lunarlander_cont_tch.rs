use anyhow::Result;
use border_core::{
    generic_replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::{AggregateRecorder, Record},
    Agent, Configurable, DefaultEvaluator, Env as _, Evaluator as _, ReplayBufferBase,
    StepProcessor, Trainer, TrainerConfig,
};
use border_derive::BatchBase;
use border_py_gym_env::{
    util::{arrayd_to_tensor, tensor_to_arrayd},
    ArrayObsFilter, ContinuousActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use border_tch_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, Sac, SacConfig},
    TensorBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::Parser;
//use csv::WriterBuilder;
use border_mlflow_tracking::MlflowTrackingClient;
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;
use std::convert::TryFrom;
use tch::Tensor;

const DIM_OBS: i64 = 8;
const DIM_ACT: i64 = 2;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 128;
const WARMUP_PERIOD: usize = 1000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 200_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const MODEL_DIR: &str = "./border/examples/gym/model/tch/sac_lunarlander_cont";

fn cuda_if_available() -> tch::Device {
    tch::Device::cuda_if_available()
}

mod obs_act_types {
    use super::*;

    type PyObsDtype = f32;

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

    #[derive(Clone, BatchBase)]
    pub struct ObsBatch(TensorBatch);

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorBatch::from_tensor(tensor))
        }
    }

    #[derive(Clone, Debug)]
    pub struct Act(ArrayD<f32>);

    impl border_core::Act for Act {}

    impl From<Act> for ArrayD<f32> {
        fn from(value: Act) -> Self {
            value.0
        }
    }

    impl From<Tensor> for Act {
        fn from(t: Tensor) -> Self {
            Self(tensor_to_arrayd(t, true))
        }
    }

    // Required by Sac
    impl From<Act> for Tensor {
        fn from(value: Act) -> Self {
            arrayd_to_tensor::<_, f32>(value.0, true)
        }
    }

    #[derive(BatchBase)]
    pub struct ActBatch(TensorBatch);

    impl From<Act> for ActBatch {
        fn from(act: Act) -> Self {
            let tensor = act.into();
            Self(TensorBatch::from_tensor(tensor))
        }
    }

    pub type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
    pub type ActFilter = ContinuousActFilter<Act>;
    pub type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Evaluator = DefaultEvaluator<Env, Sac<Env, Mlp, Mlp2, ReplayBuffer>>;
}

use obs_act_types::*;

mod config {
    use serde::Serialize;

    use super::*;

    #[derive(Serialize)]
    pub struct SacLunarLanderConfig {
        pub trainer_config: TrainerConfig,
        pub replay_buffer_config: SimpleReplayBufferConfig,
        pub agent_config: SacConfig<Mlp, Mlp2>,
    }

    pub fn env_config() -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name("LunarLanderContinuous-v2".to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn trainer_config(max_opts: usize, eval_interval: usize) -> TrainerConfig {
        TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(eval_interval)
            .record_agent_info_interval(EVAL_INTERVAL)
            .record_compute_cost_interval(EVAL_INTERVAL)
            .flush_record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .warmup_period(WARMUP_PERIOD)
            .model_dir(MODEL_DIR)
    }

    pub fn agent_config(in_dim: i64, out_dim: i64) -> SacConfig<Mlp, Mlp2> {
        let device = cuda_if_available();
        let actor_config = ActorConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
            .out_dim(out_dim)
            .pi_config(MlpConfig::new(in_dim, vec![64, 64], out_dim, false));
        let critic_config = CriticConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
            .q_config(MlpConfig::new(in_dim + out_dim, vec![64, 64], 1, false));

        SacConfig::default()
            .batch_size(BATCH_SIZE)
            .actor_config(actor_config)
            .critic_config(critic_config)
            .device(device)
    }
}

#[derive(Debug, Serialize)]
struct LunarlanderRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f32>,
    act: Vec<f32>,
}

impl TryFrom<&Record> for LunarlanderRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record.get_array1("obs")?.to_vec(),
            act: record.get_array1("act")?.to_vec(),
        })
    }
}

mod utils {
    use super::*;

    pub fn create_recorder(
        args: &Args,
        config: &config::SacLunarLanderConfig,
    ) -> Result<Box<dyn AggregateRecorder>> {
        match args.mlflow {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("Gym")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", "lunarlander")?;
                recorder_run.set_tag("algo", "sac")?;
                recorder_run.set_tag("backend", "tch")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(MODEL_DIR))),
        }
    }
}

/// Train/eval SAC agent in lunarlander environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Train SAC agent, not evaluate
    #[arg(short, long, default_value_t = false)]
    train: bool,

    /// Evaluate SAC agent, not train
    #[arg(short, long, default_value_t = false)]
    eval: bool,

    /// Log metrics with MLflow
    #[arg(short, long, default_value_t = false)]
    mlflow: bool,
}

fn train(args: &Args, max_opts: usize) -> Result<()> {
    let env_config = config::env_config();
    let trainer_config = config::trainer_config(max_opts, EVAL_INTERVAL);
    let step_proc_config = SimpleStepProcessorConfig {};
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let agent_config = config::agent_config(DIM_OBS, DIM_ACT);
    let config = config::SacLunarLanderConfig {
        agent_config: agent_config.clone(),
        replay_buffer_config: replay_buffer_config.clone(),
        trainer_config,
    };
    let mut recorder = utils::create_recorder(&args, &config)?;
    let mut trainer = Trainer::build(config.trainer_config.clone());

    let env = Env::build(&env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut agent = Sac::build(config.agent_config);
    let mut buffer = ReplayBuffer::build(&replay_buffer_config);
    let mut evaluator = Evaluator::new(&env_config, 0, N_EPISODES_PER_EVAL)?;

    trainer.train(
        env,
        step_proc,
        &mut agent,
        &mut buffer,
        &mut recorder,
        &mut evaluator,
    )?;

    Ok(())
}

fn eval(render: bool) -> Result<()> {
    let model_dir = MODEL_DIR.to_owned() + "/best";
    let env_config = {
        let mut env_config = config::env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        }
        env_config
    };
    let mut agent = {
        let mut agent = Sac::build(config::agent_config(DIM_OBS, DIM_ACT));
        agent.load_params(model_dir)?;
        agent.eval();
        agent
    };

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let args = Args::parse();

    if args.eval {
        eval(true)?;
    } else if args.train {
        train(&args, MAX_OPTS)?;
    } else {
        train(&args, MAX_OPTS)?;
        eval(true)?;
    }

    Ok(())
}
