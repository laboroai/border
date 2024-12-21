use anyhow::Result;
use border_async_trainer::{
    util::train_async, /*ActorManager as ActorManager_,*/ ActorManagerConfig,
    /*AsyncTrainer as AsyncTrainer_,*/ AsyncTrainerConfig,
};
use border_core::{
    generic_replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::Recorder,
    DefaultEvaluator,
};
use border_derive::BatchBase;
use border_mlflow_tracking::MlflowTrackingClient;
use border_py_gym_env::{
    ArrayObsFilter, ContinuousActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use border_tch_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, EntCoefMode, Sac, SacConfig},
    util::CriticLoss,
    util::{arrayd_to_tensor, tensor_to_arrayd},
    TensorBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::Parser;
use ndarray::{ArrayD, IxDyn};
use std::{convert::TryFrom, default::Default};
use tch::Tensor;

const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const WARMUP_PERIOD: usize = 10_000;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 5_000;
const SAVE_INTERVAL: usize = 500_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const N_CRITICS: usize = 2;
const TAU: f64 = 0.02;
const LR_ENT_COEF: f64 = 3e-4;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const MODEL_DIR_BASE: &str = "./border/examples/mujoco/model/tch";

fn cuda_if_available() -> tch::Device {
    tch::Device::cuda_if_available()
}

mod obs_act_types {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct Obs(ArrayD<f32>);

    #[derive(Clone, BatchBase)]
    pub struct ObsBatch(TensorBatch);

    impl border_core::Obs for Obs {
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

    type PyObsDtype = f32;
    pub type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
    pub type ActFilter = ContinuousActFilter<Act>;
    pub type EnvConfig = GymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
    pub type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Evaluator = DefaultEvaluator<Env, Sac<Env, Mlp, Mlp2, ReplayBuffer>>;
}

use obs_act_types::*;

mod config {
    use serde::Serialize;

    use super::*;

    pub fn env_config(env_name: &str) -> EnvConfig {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name(env_name.to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn create_async_trainer_config(model_dir: &str) -> Result<AsyncTrainerConfig> {
        Ok(AsyncTrainerConfig::default()
            .model_dir(model_dir)?
            .max_opts(MAX_OPTS)?
            .eval_interval(EVAL_INTERVAL)?
            .flush_record_interval(EVAL_INTERVAL)?
            .record_compute_cost_interval(EVAL_INTERVAL)?
            .sync_interval(1)?
            .save_interval(SAVE_INTERVAL)?
            .warmup_period(WARMUP_PERIOD)?)
    }

    pub fn create_sac_config(dim_obs: i64, dim_act: i64, target_ent: f64) -> SacConfig<Mlp, Mlp2> {
        let device = cuda_if_available();
        let actor_config = ActorConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
            .out_dim(dim_act)
            .pi_config(MlpConfig::new(dim_obs, vec![400, 300], dim_act, false));
        let critic_config = CriticConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
            .q_config(MlpConfig::new(dim_obs + dim_act, vec![400, 300], 1, false));

        SacConfig::default()
            .batch_size(BATCH_SIZE)
            .actor_config(actor_config)
            .critic_config(critic_config)
            .tau(TAU)
            .critic_loss(CRITIC_LOSS)
            .n_critics(N_CRITICS)
            .ent_coef_mode(EntCoefMode::Auto(target_ent, LR_ENT_COEF))
            .device(device)
    }

    pub fn show_config(
        env_config: &EnvConfig,
        agent_config: &SacConfig<Mlp, Mlp2>,
        actor_man_config: &ActorManagerConfig,
        trainer_config: &AsyncTrainerConfig,
    ) {
        println!("Device: {:?}", tch::Device::cuda_if_available());
        println!("{}", serde_yaml::to_string(&env_config).unwrap());
        println!("{}", serde_yaml::to_string(&agent_config).unwrap());
        println!("{}", serde_yaml::to_string(&actor_man_config).unwrap());
        println!("{}", serde_yaml::to_string(&trainer_config).unwrap());
    }

    #[derive(Serialize)]
    pub struct SacMujocoAsyncConfig {
        pub trainer: AsyncTrainerConfig,
        pub replay_buffer: SimpleReplayBufferConfig,
        pub agent: SacConfig<Mlp, Mlp2>,
    }
}

mod utils {
    use super::*;

    pub fn model_dir(args: &Args) -> String {
        let name = &args.env;
        format!("./border/examples/mujoco/model/tch/sac_{}_async", name)
    }

    pub fn create_recorder(
        args: &Args,
        config: &config::SacMujocoAsyncConfig,
    ) -> Result<Box<dyn Recorder>> {
        let env_name = &args.env;
        let (_, _, _, _, model_dir) = env_params(&args);
        match args.mlflow {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment("Gym")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", env_name)?;
                recorder_run.set_tag("algo", "sac")?;
                recorder_run.set_tag("backend", "tch")?;
                recorder_run.set_tag("n_actors", args.n_actors.to_string())?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }

    /// Returns (dim_obs, dim_act, target_ent, env_name, model_dir)
    pub fn env_params<'a>(args: &Args) -> (i64, i64, f64, &'a str, String) {
        let env_name = &args.env;
        let model_dir = format!("{}/sac_{}", MODEL_DIR_BASE, env_name);
        match args.env.as_str() {
            "ant" => (27, 8, -8., "Ant-v4", model_dir),
            "cheetah" => (17, 6, -6., "HalfCheetah-v4", model_dir),
            "walker" => (17, 6, -6., "Walker2d-v4", model_dir),
            "hopper" => (11, 3, -3., "Hopper-v4", model_dir),
            env => panic!("Unsupported env {:?}", env),
        }
    }
}

/// Train SAC agent in Mujoco environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Name of the environment
    env: String,

    /// Create config files
    #[arg(long, default_value_t = false)]
    create_config: bool,

    /// Show config
    #[arg(long, default_value_t = false)]
    show_config: bool,

    /// Log metrics with MLflow
    #[arg(long, default_value_t = false)]
    mlflow: bool,

    /// Waiting time in milliseconds between frames when evaluation
    #[arg(long, default_value_t = 25)]
    wait: u64,

    /// Number of actors, default to 6
    #[arg(long, default_value_t = 6)]
    n_actors: usize,
}

fn train(args: &Args) -> Result<()> {
    let (dim_obs, dim_act, target_ent, env_name, _model_dir) = utils::env_params(args);
    let env_config_train = config::env_config(env_name);
    let model_dir = utils::model_dir(&args);
    let n_actors = args.n_actors;

    // Configurations
    let agent_config =
        config::create_sac_config(dim_obs, dim_act, target_ent).device(cuda_if_available());
    let agent_configs = (0..n_actors)
        .map(|_| agent_config.clone().device(tch::Device::Cpu))
        .collect::<Vec<_>>();
    let env_config_eval = config::env_config(env_name);
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let step_proc_config = SimpleStepProcessorConfig::default();
    let actor_man_config = ActorManagerConfig::default();
    let async_trainer_config = config::create_async_trainer_config(model_dir.as_str())?;

    if args.show_config {
        config::show_config(
            &env_config_train,
            &agent_config,
            &actor_man_config,
            &async_trainer_config,
        );
    } else {
        let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;
        let config = config::SacMujocoAsyncConfig {
            trainer: async_trainer_config.clone(),
            replay_buffer: replay_buffer_config.clone(),
            agent: agent_config.clone(),
        };
        let mut recorder = utils::create_recorder(&args, &config)?;

        train_async::<_, Env, ReplayBuffer, StepProc>(
            &agent_config,
            &agent_configs,
            &env_config_train,
            &env_config_eval,
            &step_proc_config,
            &replay_buffer_config,
            &actor_man_config,
            &async_trainer_config,
            &mut recorder,
            &mut evaluator,
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    tch::set_num_threads(1);
    let args = Args::parse();
    train(&args)?;
    Ok(())
}
