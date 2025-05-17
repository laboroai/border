use anyhow::Result;
use border_candle_agent::{
    awac::{Awac, AwacConfig},
    mlp::{Mlp, Mlp3, MlpConfig},
    opt::OptimizerConfig,
    util::{
        actor::{ActionLimit, GaussianActorConfig},
        critic::MultiCriticConfig,
    },
    Activation,
};
use border_core::{
    generic_replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::Recorder,
    Agent, Configurable, DefaultEvaluator, Env as _, Evaluator as _, ReplayBufferBase,
    StepProcessor, Trainer, TrainerConfig,
};
use border_mlflow_tracking::MlflowTrackingClient;
use border_py_gym_env::{
    candle::{NdarrayConverter, NdarrayConverterConfig, TensorBatch},
    GymEnv, GymEnvConfig,
};
use border_tensorboard::TensorboardRecorder;
use candle_core::Device;
use clap::Parser;
use serde::Serialize;

type Env = GymEnv<NdarrayConverter>;
type ReplayBuffer = SimpleReplayBuffer<TensorBatch, TensorBatch>;
type StepProc = SimpleStepProcessor<Env, TensorBatch, TensorBatch>;
type Evaluator = DefaultEvaluator<Env>;

const DIM_OBS: i64 = 3;
const DIM_ACT: i64 = 1;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 128;
const WARMUP_PERIOD: usize = 1000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 40_000;
const EVAL_INTERVAL: usize = 2_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const ENV_NAME: &str = "Pendulum-v1";
const MODEL_DIR: &str = "./model/candle/awac_pendulum";
const MLFLOW_EXPERIMENT_NAME: &str = "Gym";
const MLFLOW_RUN_NAME: &str = "awac-gym-pendulum-v1-candle";
const MLFLOW_TAGS: &[(&str, &str)] =
    &[("env", "pendulum"), ("algo", "awac"), ("backend", "candle")];

/// Train/eval Awac agent in pendulum environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Train Awac agent, not evaluate
    #[arg(short, long, default_value_t = false)]
    train: bool,

    /// Evaluate Awac agent, not train
    #[arg(short, long, default_value_t = false)]
    eval: bool,

    /// Log metrics with MLflow
    #[arg(short, long, default_value_t = false)]
    mlflow: bool,
}

fn create_env_config(render: bool) -> Result<GymEnvConfig<NdarrayConverter>> {
    let mut env_config = GymEnvConfig::default()
        .name(ENV_NAME.to_string())
        .converter_config(NdarrayConverterConfig {});

    if render {
        env_config = env_config
            .render_mode(Some("human".to_string()))
            .set_wait_in_millis(10);
    }

    Ok(env_config)
}

fn create_awac_config(in_dim: i64, out_dim: i64) -> Result<AwacConfig<Mlp, Mlp3>> {
    // Dimensions of observation and action
    let dim_obs = in_dim;
    let dim_act = out_dim;

    // Actor/Critic learning rate
    let lr = 0.0003;

    // Actor/critic configs
    let actor_config = GaussianActorConfig::default()
        .opt_config(OptimizerConfig::Adam { lr })
        .out_dim(dim_act)
        .action_limit(ActionLimit::Tanh { action_scale: 1.0 })
        .policy_config(MlpConfig::new(
            dim_obs,
            vec![256, 256, 256],
            dim_act,
            Activation::None,
        ));
    let critic_config = MultiCriticConfig::default()
        .opt_config(OptimizerConfig::Adam { lr })
        .q_config(MlpConfig::new(
            dim_obs + dim_act,
            vec![256, 256, 256],
            1,
            Activation::None,
        ));

    // Device
    let device = Device::cuda_if_available(0)?;

    // Agent config
    let agent_config = AwacConfig::<Mlp, Mlp3>::default()
        .actor_config(actor_config)
        .critic_config(critic_config)
        .device(device)
        .batch_size(BATCH_SIZE);
    Ok(agent_config)
}

/// `model_dir` - Directory where TFRecord and model parameters are saved with
///               [`TensorboardRecorder`].
/// `config` - Configuration parameters for a run of MLflow. These are used for
///            recording purpose only when a new run is created.
fn create_recorder(
    args: &Args,
    model_dir: &str,
    config: Option<&AwacPendulumConfig>,
) -> Result<Box<dyn Recorder<Env, ReplayBuffer>>> {
    match args.mlflow {
        true => {
            let client = MlflowTrackingClient::new("http://localhost:8080")
                .set_experiment(MLFLOW_EXPERIMENT_NAME)?;
            let recorder_run = client.create_recorder(MLFLOW_RUN_NAME)?;
            if let Some(config) = config {
                recorder_run.log_params(config)?;
                recorder_run.set_tags(MLFLOW_TAGS)?;
            }
            Ok(Box::new(recorder_run))
        }
        false => Ok(Box::new(TensorboardRecorder::new(
            model_dir, model_dir, false,
        ))),
    }
}

#[derive(Serialize)]
pub struct AwacPendulumConfig {
    pub env_config: GymEnvConfig<NdarrayConverter>,
    pub agent_config: AwacConfig<Mlp, Mlp3>,
    pub trainer_config: TrainerConfig,
}

impl AwacPendulumConfig {
    pub fn new(in_dim: i64, out_dim: i64, max_opts: usize, eval_interval: usize) -> Result<Self> {
        let env_config = create_env_config(false)?;
        let agent_config = create_awac_config(in_dim, out_dim)?;
        let trainer_config = TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(eval_interval)
            .record_agent_info_interval(EVAL_INTERVAL)
            .record_compute_cost_interval(EVAL_INTERVAL)
            .flush_record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .warmup_period(WARMUP_PERIOD);
        let config = Self {
            env_config,
            agent_config,
            trainer_config,
        };

        Ok(config)
    }
}

fn train(args: &Args, max_opts: usize, model_dir: &str, eval_interval: usize) -> Result<()> {
    let config = AwacPendulumConfig::new(DIM_OBS, DIM_ACT, max_opts, eval_interval)?;
    let step_proc_config = SimpleStepProcessorConfig {};
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let mut recorder = create_recorder(&args, model_dir, Some(&config))?;
    let mut trainer = Trainer::build(config.trainer_config.clone());

    let env = Env::build(&config.env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut agent = Box::new(Awac::build(config.agent_config)) as _;
    let mut buffer = ReplayBuffer::build(&replay_buffer_config);
    let mut evaluator = Evaluator::new(&config.env_config, 0, N_EPISODES_PER_EVAL)?;

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

fn eval(args: &Args, model_dir: &str, render: bool) -> Result<()> {
    let env_config = create_env_config(render)?;
    let mut agent: Box<dyn Agent<_, ReplayBuffer>> = {
        let agent_config = create_awac_config(DIM_OBS, DIM_ACT)?;
        let mut agent = Box::new(Awac::build(agent_config)) as _;
        let recorder = create_recorder(&args, model_dir, None)?;
        recorder.load_model("best".as_ref(), &mut agent)?;
        agent.eval();
        agent
    };
    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if args.train {
        train(&args, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
    } else if args.eval {
        eval(&args, MODEL_DIR, true)?;
    } else {
        train(&args, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
        eval(&args, MODEL_DIR, true)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_awac_pendulum() -> Result<()> {
        let tmp_dir = TempDir::new("awac_pendulum")?;
        let model_dir = match tmp_dir.as_ref().to_str() {
            Some(s) => s,
            None => panic!("Failed to get string of temporary directory"),
        };
        let args = Args {
            train: false,
            eval: false,
            mlflow: false,
        };
        train(&args, 100, model_dir, 100)?;
        eval(&args, model_dir, false)?;
        Ok(())
    }
}
