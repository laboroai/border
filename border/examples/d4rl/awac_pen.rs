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

const MODEL_DIR: &str = "border/examples/d4rl/model/candle/awac_pen";
const MLFLOW_EXPERIMENT_NAME: &str = "D4RL";
const MLFLOW_TAGS: &[(&str, &str)] = &[("algo", "awac"), ("backend", "candle")];

/// Train BC agent in pen environment
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
#[command(version, about)]
struct Args {
    /// "train" or "eval".
    /// In evaluation mode, the trained model is loaded.
    #[arg(long)]
    mode: String,

    /// Name of environment ID, e.g., human-v2.
    /// See Minari documantation:
    /// https://minari.farama.org/v0.5.1/datasets/D4RL/pen/
    #[arg(long)]
    env: String,

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

    /// The number of optimization steps
    #[arg(long, default_value_t = 1000000)]
    max_opts: usize,

    /// Interval of evaluation
    #[arg(long, default_value_t = 10000)]
    eval_interval: usize,

    /// The number of evaluation episodes
    #[arg(long, default_value_t = 5)]
    eval_episodes: usize,

    /// Batch size
    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    /// Action limit type ("clamp" or "tanh")
    #[arg(long, default_value = "clamp")]
    action_limit: String,
}

impl Args {
    pub fn env_name(&self) -> String {
        format!("pen/{}", self.env)
    }

    pub fn dataset_name(&self) -> String {
        format!("D4RL/pen/{}", self.env)
    }

    pub fn action_limit(&self) -> ActionLimit {
        match self.action_limit.as_str() {
            "clamp" => ActionLimit::Clamp {
                action_min: -1.0,
                action_max: 1.0,
            },
            "tanh" => ActionLimit::Tanh { action_scale: 1.0 },
            _ => panic!("action_limit should be clamp or tanh"),
        }
    }
}

#[derive(Serialize)]
struct PenConfig {
    args: Args,
    trainer_config: TrainerConfig,
    agent_config: AwacConfig<Mlp, Mlp3>,
}

impl PenConfig {
    fn new(args: Args) -> Self {
        let trainer_config = TrainerConfig::default()
            .max_opts(args.max_opts)
            .eval_interval(args.eval_interval)
            .flush_record_interval(args.eval_interval)
            .record_agent_info_interval(args.eval_interval);
        let agent_config = create_awac_config(&args).unwrap();
        Self {
            args,
            trainer_config,
            agent_config,
        }
    }
}

fn create_awac_config(args: &Args) -> Result<AwacConfig<Mlp, Mlp3>> {
    // Dimensions of observation and action
    let dim_obs = 45;
    let dim_act = 24;

    // Actor/Critic learning rate
    let lr = 0.0003;

    // Actor/critic configs
    let actor_config = GaussianActorConfig::default()
        .opt_config(OptimizerConfig::default().learning_rate(lr))
        .out_dim(dim_act)
        .action_limit(args.action_limit())
        .policy_config(MlpConfig::new(
            dim_obs,
            vec![256, 256],
            dim_act,
            Activation::None,
        ));
    let critic_config = MultiCriticConfig::default()
        .opt_config(OptimizerConfig::default().learning_rate(lr))
        .q_config(MlpConfig::new(
            dim_obs + dim_act,
            vec![256, 256],
            1,
            Activation::None,
        ));

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
    let agent_config = AwacConfig::<Mlp, Mlp3>::default()
        .actor_config(actor_config)
        .critic_config(critic_config)
        .device(device)
        .batch_size(args.batch_size)
        .lambda(0.3333)
        .adv_softmax(true);
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
    Box::new(Awac::build(config.agent_config.clone()))
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
        Ok(Box::new(TensorboardRecorder::new(
            MODEL_DIR, MODEL_DIR, false,
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
