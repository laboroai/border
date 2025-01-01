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
    d4rl::pointmaze::{
        candle::{PointMazeConverter, PointMazeConverterConfig},
        PointMazeEvaluator,
    },
    MinariConverter, MinariDataset, MinariEnv,
};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tensorboard::TensorboardRecorder;
use candle_core::{Device, Tensor};
use clap::Parser;
use serde::Serialize;
use std::fmt::Debug;

/// Train BC agent in maze2d environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    // /// Environment name (ant, cheetah, walker, hopper)
    // #[arg(long)]
    // env: String,

    // /// Train DQN agent, not evaluate
    // #[arg(long, default_value_t = false)]
    // train: bool,

    // /// Evaluate DQN agent, not train
    // #[arg(long, default_value_t = false)]
    // eval: bool,
    // #[arg(long, default_value_t = String::new())]
    // eval: String,

    // /// Waiting time in milliseconds between frames when evaluation
    // #[arg(long, default_value_t = 25)]
    // wait: u64,
    /// Log metrics with MLflow
    #[arg(long, default_value_t = false)]
    mlflow: bool,

    /// The number of optimization steps
    #[arg(long, default_value_t = 1000000)]
    max_opts: usize,

    /// Interval of evaluation
    #[arg(long, default_value_t = 100000)]
    eval_interval: usize,

    /// The number of evaluation episodes
    #[arg(long, default_value_t = 5)]
    eval_episodes: usize,

    /// If true, goal position is included in observation
    #[arg(long, default_value_t = false)]
    include_goal: bool,

    /// Batch size
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
}

const MODEL_DIR: &str = "border/examples/d4rl/model/candle/bc_maze2d";

fn create_trainer(args: &Args) -> Trainer {
    log::info!("Create trainer");
    Trainer::build(
        TrainerConfig::default()
            .max_opts(args.max_opts)
            .eval_interval(args.eval_interval)
            .flush_record_interval(args.max_opts / 50)
            .record_agent_info_interval(args.max_opts / 50),
    )
}

fn create_agent<E, R>(args: &Args) -> (Box<dyn Agent<E, R>>, impl Serialize)
where
    E: Env + 'static,
    E::Obs: Into<Tensor>,
    E::Act: From<Tensor> + Into<Tensor>,
    R: ReplayBufferBase + 'static,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Tensor>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    // Dimensions of observation and action
    let dim_obs = match args.include_goal {
        true => 6,
        false => 4,
    };
    let dim_act = 2;

    log::info!("Create agent");
    let agent_config = {
        let policy_model_config = {
            let policy_model_config = MlpConfig {
                in_dim: dim_obs,
                out_dim: dim_act,
                units: vec![256, 256],
                activation_out: Activation::Tanh,
            };
            BcModelConfig::default().policy_model_config(policy_model_config)
        };
        BcConfig::<Mlp>::default()
            .policy_model_config(policy_model_config)
            .batch_size(args.batch_size)
            .device(Device::Cpu)
            .action_type(BcActionType::Continuous)
            .optimizer(border_candle_agent::opt::OptimizerConfig::AdamW {
                lr: 0.0003,
                beta1: candle_nn::ParamsAdamW::default().beta1,
                beta2: candle_nn::ParamsAdamW::default().beta2,
                eps: candle_nn::ParamsAdamW::default().eps,
                weight_decay: candle_nn::ParamsAdamW::default().weight_decay,
            })
    };

    (Box::new(Bc::build(agent_config.clone())), agent_config)
}

fn create_replay_buffer<T>(
    converter: &T,
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

fn create_recorder<E, R>(
    args: &Args,
    agent_config: &impl Serialize,
) -> Result<Box<dyn Recorder<E, R>>>
where
    E: Env + 'static,
    R: ReplayBufferBase + 'static,
{
    log::info!("Create recorder");
    match args.mlflow {
        true => {
            let client =
                MlflowTrackingClient::new("http://localhost:8080").set_experiment("D4RL")?;
            let recorder_run = client.create_recorder("")?;
            recorder_run.log_params(&agent_config)?;
            recorder_run.set_tag("env", "maze2d")?;
            recorder_run.set_tag("algo", "bc")?;
            recorder_run.set_tag("backend", "candle")?;
            Ok(Box::new(recorder_run))
        }
        false => Ok(Box::new(TensorboardRecorder::new(
            MODEL_DIR, MODEL_DIR, false,
        ))),
    }
}

fn create_evaluator<T>(
    args: &Args,
    converter: T,
    dataset: &MinariDataset,
) -> Result<impl Evaluator<MinariEnv<T>>>
where
    T: MinariConverter,
{
    // Create evaluator
    log::info!("Create evaluator");
    let env = dataset.recover_environment(converter, true, None)?;
    PointMazeEvaluator::new(env, args.eval_episodes)
}

fn train<T>(args: Args, dataset: MinariDataset, converter: T) -> Result<()>
where
    T: MinariConverter + 'static,
    T::Obs: std::fmt::Debug + Into<Tensor>,
    T::Act: std::fmt::Debug + From<Tensor> + Into<Tensor>,
    T::ObsBatch: std::fmt::Debug + Into<Tensor> + 'static,
    T::ActBatch: std::fmt::Debug + Into<Tensor> + 'static,
{
    let mut trainer = create_trainer(&args);
    let (mut agent, agent_config) = create_agent(&args);
    let mut buffer = create_replay_buffer(&converter, &dataset)?;
    let mut recorder = create_recorder(&args, &agent_config)?;
    let mut evaluator = create_evaluator(&args, converter, &dataset)?;

    log::info!("Start training");
    let _ = trainer.train_offline(&mut agent, &mut buffer, &mut recorder, &mut evaluator);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let dataset = MinariDataset::load_dataset("D4RL/pointmaze/umaze-v2", true)?;
    let converter = PointMazeConverter::new(PointMazeConverterConfig {
        // Include goal position in observation
        include_goal: args.include_goal,
    });

    train(args, dataset, converter)
}
