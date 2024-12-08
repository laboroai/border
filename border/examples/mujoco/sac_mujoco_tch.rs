use anyhow::Result;
// use border::util::get_model_from_url;
use border_core::{
    generic_replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::AggregateRecorder,
    Agent, Configurable, DefaultEvaluator, Env as _, Evaluator as _, ReplayBufferBase,
    StepProcessor, Trainer, TrainerConfig,
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
// use log::info;
use ndarray::{ArrayD, IxDyn};
use std::convert::TryFrom;
use tch::Tensor;

const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const WARMUP_PERIOD: usize = 10_000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 5_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const N_EPISODES_PER_EVAL: usize = 5;
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

    #[derive(Serialize)]
    pub struct SacAntConfig {
        pub trainer: TrainerConfig,
        pub replay_buffer: SimpleReplayBufferConfig,
        pub agent: SacConfig<Mlp, Mlp2>,
    }

    pub fn env_config(env_name: &str) -> EnvConfig {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name(env_name.to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn create_trainer_config(model_dir: &str) -> TrainerConfig {
        TrainerConfig::default()
            .max_opts(MAX_OPTS)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(EVAL_INTERVAL)
            .record_agent_info_interval(EVAL_INTERVAL)
            .record_compute_cost_interval(EVAL_INTERVAL)
            .flush_record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .warmup_period(WARMUP_PERIOD)
            .model_dir(model_dir)
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
}

mod utils {
    use super::*;

    pub fn create_recorder(
        args: &Args,
        config: &config::SacAntConfig,
    ) -> Result<Box<dyn AggregateRecorder>> {
        let env_name = &args.env;
        let (_, _, _, _, model_dir) = env_params(&args);
        match args.mlflow {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("Gym")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", env_name)?;
                recorder_run.set_tag("algo", "sac")?;
                recorder_run.set_tag("backend", "tch")?;
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

/// Train/eval SAC agent in Mujoco environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Environment name (ant, cheetah, walker, hopper)
    #[arg(long)]
    env: String,

    /// Train DQN agent, not evaluate
    #[arg(long, default_value_t = false)]
    train: bool,

    /// Evaluate DQN agent, not train
    #[arg(long, default_value_t = false)]
    eval: bool,
    // #[arg(long, default_value_t = String::new())]
    // eval: String,
    /// Log metrics with MLflow
    #[arg(long, default_value_t = false)]
    mlflow: bool,

    /// Waiting time in milliseconds between frames when evaluation
    #[arg(long, default_value_t = 25)]
    wait: u64,
}

fn train(args: &Args) -> Result<()> {
    let (dim_obs, dim_act, target_ent, env_name, model_dir) = utils::env_params(args);
    let env_config = config::env_config(env_name);
    let step_proc_config = SimpleStepProcessorConfig {};
    let agent_config = config::create_sac_config(dim_obs, dim_act, target_ent);
    let trainer_config = config::create_trainer_config(&model_dir);
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let mut trainer = Trainer::build(trainer_config.clone());

    let config = config::SacAntConfig {
        trainer: trainer_config,
        replay_buffer: replay_buffer_config.clone(),
        agent: agent_config.clone(),
    };
    let env = Env::build(&env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut agent = Sac::build(agent_config);
    let mut buffer = ReplayBuffer::build(&replay_buffer_config);
    let mut recorder = utils::create_recorder(args, &config)?;
    let mut evaluator = {
        let env = Env::build(&env_config, 0)?;
        Evaluator::new(env, N_EPISODES_PER_EVAL)?
    };

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

fn eval(args: &Args, model_dir: &str, render: bool, wait: u64) -> Result<()> {
    let (dim_obs, dim_act, target_ent, env_name, _) = utils::env_params(&args);
    let env_config = {
        let mut env_config = config::env_config(&env_name);
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(wait);
        };
        env_config
    };
    let mut agent = {
        let agent_config = config::create_sac_config(dim_obs, dim_act, target_ent);
        let mut agent = Sac::build(agent_config);
        match agent.load_params(model_dir) {
            Ok(_) => {}
            Err(_) => println!("Failed to load model parameters from {:?}", model_dir),
        }
        agent.eval();
        Box::new(agent)
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = {
        let env = Env::build(&env_config, 0)?;
        Evaluator::new(env, N_EPISODES_PER_EVAL)?
    }
    .evaluate(&mut agent);

    Ok(())
}

fn eval1(args: &Args) -> Result<()> {
    let model_dir = {
        let env_name = &args.env;
        format!("{}/sac_{}/best", MODEL_DIR_BASE, env_name)
    };
    let render = true;
    let wait = args.wait;
    eval(&args, &model_dir, render, wait)
}

// fn eval2(matches: ArgMatches) -> Result<()> {
//     let model_dir = {
//         let file_base = "sac_ant_20210324_ec2_smoothl1";
//         let url =
//             "https://drive.google.com/uc?export=download&id=1XvFi2nJD5OhpTvs-Et3YREuoqy8c3Vkq";
//         let model_dir = get_model_from_url(url, file_base)?;
//         info!("Download the model in {:?}", model_dir.as_ref().to_str());
//         model_dir.as_ref().to_str().unwrap().to_string()
//     };
//     let render = true;
//     let wait = matches.value_of("wait").unwrap().parse().unwrap();
//     eval(&matches, &model_dir, render, wait)
// }

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let args = Args::parse();

    if args.train {
        train(&args)?;
    } else if args.eval {
        eval1(&args)?;
    } else {
        train(&args)?;
        eval1(&args)?;
    }
    // } else if matches.is_present("play-gdrive") {
    //     eval2(matches)?;

    Ok(())
}
