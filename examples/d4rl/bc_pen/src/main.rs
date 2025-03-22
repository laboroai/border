use anyhow::Result;
use border_candle_agent::{
    bc::{Bc, BcActionType, BcConfig, BcModelConfig},
    mlp::{Mlp, MlpConfig},
    Activation,
};
use border_core::{
    generic_replay_buffer::{BatchBase, SimpleReplayBuffer},
    record::Recorder,
    Agent, Configurable, Env, Evaluator, ExperienceBufferBase, ReplayBufferBase, Trainer,
    TrainerConfig, TransitionBatch,
};
use border_minari::{
    d4rl::pen::candle::{PenConverter, PenConverterConfig},
    MinariConverter, MinariDataset, MinariEnv, MinariEvaluator,
};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tensorboard::TensorboardRecorder;
use candle_core::{Device, Tensor};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

const MODEL_DIR: &str = "./model";
const MLFLOW_EXPERIMENT_NAME: &str = "D4RL";
const MLFLOW_TAGS: &[(&str, &str)] = &[("algo", "bc"), ("backend", "candle")];

/// Train BC agent in pen environment
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
#[command(version, about)]
struct Args {
    /// "train" or "eval".
    /// In evaluation mode, the trained model is loaded.
    #[arg(long)]
    mode: String,

    /// Device name.
    /// If set to `"Cpu"`, the CPU will be used.
    /// Otherwise, the device will be determined by the `cuda_if_available()` method.
    #[arg(long)]
    device: Option<String>,

    // /// Waiting time in milliseconds between frames when evaluation
    // #[arg(long, default_value_t = 25)]
    // wait: u64,
    /// Run name of MLflow.
    /// When using this option, an MLflow server must be running.
    /// If no name is provided, the log will be recorded in TensorBoard.
    #[arg(long)]
    mlflow_run_name: Option<String>,

    /// Name of environment ID, e.g., human-v2.
    /// See Minari documantation:
    /// https://minari.farama.org/v0.5.1/datasets/D4RL/pen/
    #[arg(long)]
    env: String,

    /// The number of optimization steps
    #[arg(long, default_value_t = 1000000)]
    max_opts: usize,

    /// Interval of evaluation
    #[arg(long, default_value_t = 1000)]
    eval_interval: usize,

    // Interval of recording agent info
    #[arg(long, default_value_t = 100)]
    record_agent_info_interval: usize,

    /// The number of evaluation episodes
    #[arg(long, default_value_t = 5)]
    eval_episodes: usize,

    /// Batch size
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
}

impl Args {
    pub fn env_name(&self) -> String {
        format!("pen/{}", self.env)
    }

    pub fn dataset_name(&self) -> String {
        format!("D4RL/pen/{}", self.env)
    }
}

#[derive(Serialize)]
struct PenConfig {
    args: Args,
    trainer_config: TrainerConfig,
    agent_config: BcConfig<Mlp>,
}

impl PenConfig {
    fn new(args: Args) -> Self {
        let trainer_config = TrainerConfig::default()
            .max_opts(args.max_opts)
            .eval_interval(args.eval_interval)
            .flush_record_interval(args.eval_interval)
            .record_agent_info_interval(args.eval_interval);
        let agent_config = create_bc_config(&args).unwrap();
        Self {
            args,
            trainer_config,
            agent_config,
        }
    }
}

fn create_bc_config(args: &Args) -> Result<BcConfig<Mlp>> {
    // Dimensions of observation and action
    let dim_obs = 45;
    let dim_act = 24;

    let policy_model_config = {
        let policy_model_config = MlpConfig {
            in_dim: dim_obs,
            out_dim: dim_act,
            units: vec![256, 256],
            activation_out: Activation::Tanh,
        };
        BcModelConfig::default().policy_model_config(policy_model_config)
    };

    // Device
    let device = if let Some(device) = &args.device {
        match device.as_str() {
            "cpu" => Device::Cpu,
            _ => Device::cuda_if_available(0)?,
        }
    } else {
        Device::cuda_if_available(0)?
    };
    log::info!("Device is {:?}", device);

    // Agent config
    let agent_config = BcConfig::<Mlp>::default()
        .policy_model_config(policy_model_config)
        .action_type(BcActionType::Continuous)
        .optimizer(border_candle_agent::opt::OptimizerConfig::AdamW {
            lr: 0.0003,
            beta1: candle_nn::ParamsAdamW::default().beta1,
            beta2: candle_nn::ParamsAdamW::default().beta2,
            eps: candle_nn::ParamsAdamW::default().eps,
            weight_decay: candle_nn::ParamsAdamW::default().weight_decay,
        })
        .device(device)
        .batch_size(args.batch_size);
    Ok(agent_config)
}

fn create_trainer(config: &PenConfig) -> Trainer {
    log::info!("Create trainer");
    Trainer::build(config.trainer_config.clone())
}

fn create_agent<E, R>(config: &PenConfig) -> Box<dyn Agent<E, R>>
where
    E: Env + 'static,
    E::Obs: Into<Tensor>,
    E::Act: From<Tensor> + Into<Tensor>,
    R: ReplayBufferBase + 'static,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Tensor> + Clone,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor> + Clone,
{
    log::info!("Create agent");
    Box::new(Bc::build(config.agent_config.clone()))
}

fn create_replay_buffer<T>(
    converter: &mut T,
    dataset: &MinariDataset,
) -> Result<SimpleReplayBuffer<T::ObsBatch, T::ActBatch>>
where
    T: MinariConverter,
    T::ObsBatch: BatchBase + Debug + Into<Tensor>,
    T::ActBatch: BatchBase + Debug + Into<Tensor>,
{
    log::info!("Create replay buffer");
    let buffer = dataset.create_replay_buffer(converter, None)?;
    log::info!("{} samples", buffer.len());
    Ok(buffer)
}

fn create_recorder<E, R>(config: &PenConfig) -> Result<Box<dyn Recorder<E, R>>>
where
    E: Env + 'static,
    R: ReplayBufferBase + 'static,
{
    log::info!("Create recorder");
    if let Some(mlflow_run_name) = &config.args.mlflow_run_name {
        let client = MlflowTrackingClient::new("http://localhost:8080")
            .set_experiment(MLFLOW_EXPERIMENT_NAME)?;
        let recorder_run = client.create_recorder(mlflow_run_name)?;
        recorder_run.log_params(config)?;
        recorder_run.set_tags(MLFLOW_TAGS)?;
        recorder_run.set_tag("env", config.args.env_name())?;
        Ok(Box::new(recorder_run))
    } else {
        let model_dir = format!("{}/{}", MODEL_DIR, config.args.env);
        Ok(Box::new(TensorboardRecorder::new(
            &model_dir, &model_dir, false,
       )))
    }
}

fn create_evaluator<T>(
    args: &Args,
    converter: T,
    dataset: &MinariDataset,
    render: bool,
) -> Result<impl Evaluator<MinariEnv<T>>>
where
    T: MinariConverter,
{
    // Create evaluator
    log::info!("Create evaluator");
    let render_mode = match render {
        true => Some("human"),
        false => None,
    };
    // for minari 0.5.1 recover pen env with eval_env=True will fail
    let env = dataset.recover_environment(converter, false, render_mode)?;
    MinariEvaluator::new(env, args.eval_episodes)
}

fn train<T>(config: PenConfig, dataset: MinariDataset, mut converter: T) -> Result<()>
where
    T: MinariConverter + 'static,
    T::Obs: std::fmt::Debug + Into<Tensor>,
    T::Act: std::fmt::Debug + From<Tensor> + Into<Tensor>,
    T::ObsBatch: std::fmt::Debug + Into<Tensor> + 'static + Clone,
    T::ActBatch: std::fmt::Debug + Into<Tensor> + 'static + Clone,
{
    let mut trainer = create_trainer(&config);
    let mut agent = create_agent(&config);
    let mut buffer = create_replay_buffer(&mut converter, &dataset)?;
    let mut recorder = create_recorder(&config)?;
    let mut evaluator = create_evaluator(&config.args, converter, &dataset, false)?;

    log::info!("Start training");
    let _ = trainer.train_offline(&mut agent, &mut buffer, &mut recorder, &mut evaluator);

    Ok(())
}

fn eval<T>(config: PenConfig, dataset: MinariDataset, converter: T) -> Result<()>
where
    T: MinariConverter + 'static,
    T::Obs: std::fmt::Debug + Into<Tensor>,
    T::Act: std::fmt::Debug + From<Tensor> + Into<Tensor>,
    T::ObsBatch: std::fmt::Debug + Into<Tensor> + 'static + Clone,
    T::ActBatch: std::fmt::Debug + Into<Tensor> + 'static + Clone,
{
    let mut agent: Box<dyn Agent<MinariEnv<T>, SimpleReplayBuffer<T::ObsBatch, T::ActBatch>>> =
        create_agent(&config);
    let recorder = create_recorder(&config)?; // used for loading a trained model
    let mut evaluator = create_evaluator(&config.args, converter, &dataset, true)?;
    recorder.load_model(Path::new("best"), &mut agent)?;
    evaluator.evaluate(&mut agent)?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let config = PenConfig::new(args.clone());
    let dataset = MinariDataset::load_dataset(args.dataset_name(), true)?;
    let converter = PenConverter::new(PenConverterConfig {}, &dataset)?;

    match args.mode.as_str() {
        "train" => train(config, dataset, converter),
        "eval" => eval(config, dataset, converter),
        _ => panic!("mode must be either 'train' or 'eval'"),
    }
}

#[test]
fn test() -> Result<()> {
    let args = Args {
        mode: "train".to_string(),
        env: "human-v2".to_string(),
        device: None,
        mlflow_run_name: None,
        max_opts: 10,
        eval_interval: 100,
        eval_episodes: 100,
        batch_size: 256,
        record_agent_info_interval: 1000,
    };
    let config = PenConfig::new(args.clone());
    let dataset = MinariDataset::load_dataset(args.dataset_name(), true)?;
    let converter = PenConverter::new(PenConverterConfig {}, &dataset)?;
    train(config, dataset, converter)
}
