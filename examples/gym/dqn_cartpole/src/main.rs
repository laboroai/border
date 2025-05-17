use anyhow::Result;
use border_candle_agent::{
    dqn::{Dqn, DqnConfig, DqnModelConfig},
    mlp::{Mlp, MlpConfig},
    opt::OptimizerConfig,
    util::CriticLoss,
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
    candle::{
        // tensor_converter::{TensorConverter, TensorConverterConfig},
        NdarrayConverter,
        NdarrayConverterConfig,
        TensorBatch,
    },
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

const DIM_OBS: i64 = 4;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 1e-3;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const WARMUP_PERIOD: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.01;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 10_000;
const EVAL_INTERVAL: usize = 1_000;
const REPLAY_BUFFER_CAPACITY: usize = 10_000;
const N_EPISODES_PER_EVAL: usize = 5;
const CRITIC_LOSS: CriticLoss = CriticLoss::Mse;
const ENV_NAME: &str = "CartPole-v0";
const MODEL_DIR: &str = "./model/candle/dqn_cartpole";
const MLFLOW_EXPERIMENT_NAME: &str = "Gym";
const MLFLOW_RUN_NAME: &str = "dqn-gym-cartpole-v0-candle";
const MLFLOW_TAGS: &[(&str, &str)] = &[("env", "cartpole"), ("algo", "dqn"), ("backend", "candle")];

/// Train/eval DQN agent in cartpole environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Train DQN agent, not evaluate
    #[arg(short, long, default_value_t = false)]
    train: bool,

    /// Evaluate DQN agent, not train
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

fn create_agent_config(in_dim: i64, out_dim: i64) -> Result<DqnConfig<Mlp>> {
    let device = Device::cuda_if_available(0)?;
    let opt_config = OptimizerConfig::default().learning_rate(LR_CRITIC);
    let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim, Activation::None);
    let model_config = DqnModelConfig::default()
        .q_config(mlp_config)
        .out_dim(out_dim)
        .opt_config(opt_config);
    let agent_config = DqnConfig::default()
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .model_config(model_config)
        .device(device)
        .critic_loss(CRITIC_LOSS);
    Ok(agent_config)
}

/// `model_dir` - Directory where TFRecord and model parameters are saved with
///               [`TensorboardRecorder`].
/// `config` - Configuration parameters for a run of MLflow. These are used for
///            recording purpose only when a new run is created.
fn create_recorder(
    args: &Args,
    model_dir: &str,
    config: Option<&DqnCartpoleConfig>,
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
pub struct DqnCartpoleConfig {
    pub env_config: GymEnvConfig<NdarrayConverter>,
    pub agent_config: DqnConfig<Mlp>,
    pub trainer_config: TrainerConfig,
}

impl DqnCartpoleConfig {
    pub fn new(in_dim: i64, out_dim: i64, max_opts: usize, eval_interval: usize) -> Result<Self> {
        let env_config = create_env_config(false)?;
        let agent_config = create_agent_config(in_dim, out_dim)?;
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
    let config = DqnCartpoleConfig::new(DIM_OBS, DIM_ACT, max_opts, eval_interval)?;
    let step_proc_config = SimpleStepProcessorConfig {};
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let mut recorder = create_recorder(&args, model_dir, Some(&config))?;
    let mut trainer = Trainer::build(config.trainer_config.clone());

    let env = Env::build(&config.env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut agent = Box::new(Dqn::build(config.agent_config)) as _;
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
        let agent_config = create_agent_config(DIM_OBS, DIM_ACT)?;
        let mut agent = Box::new(Dqn::build(agent_config)) as _;
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
mod tests {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_dqn_cartpole() -> Result<()> {
        let tmp_dir = TempDir::new("dqn_cartpole")?;
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
