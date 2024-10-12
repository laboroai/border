use anyhow::Result;
use border_candle_agent::{
    bc::{Bc, BcActionType, BcConfig, BcModelConfig},
    mlp::{Mlp, MlpConfig},
};
use border_core::{
    record::AggregateRecorder, Configurable, DefaultEvaluator, ExperienceBufferBase, Trainer,
    TrainerConfig,
};
use border_minari::{d4rl::kitchen::candle::KitchenConverter, MinariDataset};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tensorboard::TensorboardRecorder;
use candle_core::Device;
use clap::Parser;

/// Train BC agent in kitchen environment
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
    #[arg(long, default_value_t = 1000)]
    max_opts: usize,

    /// Interval of evaluation
    #[arg(long, default_value_t = 100)]
    eval_interval: usize,

    /// The number of evaluation episodes
    #[arg(long, default_value_t = 5)]
    eval_episodes: usize,

    /// Batch size
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
}

const MODEL_DIR: &str = "border/examples/d4rl/model/candle/bc_kitchen";

fn main() -> Result<()> {
    let args = Args::parse();

    let dataset = MinariDataset::load_dataset("D4RL/kitchen/complete-v1", true)?;

    // Converter for observation and action
    let converter = KitchenConverter {};

    // Create replay buffer
    let mut buffer = dataset.create_replay_buffer(&converter, None)?;
    println!(
        "Replay buffer was created with {} transitions.",
        buffer.len()
    );

    // Create environment
    let env = dataset.recover_environment(converter, false, None)?;

    // Create trainer
    let mut trainer = Trainer::build(
        TrainerConfig::default()
            .max_opts(args.max_opts)
            .eval_interval(args.eval_interval)
            .flush_record_interval(args.max_opts / 50)
            .record_agent_info_interval(args.max_opts / 50)
            .model_dir(MODEL_DIR),
    );

    // Create behavior cloning agent
    let agent_config = {
        let policy_model_config = {
            let policy_model_config = MlpConfig {
                in_dim: 59,
                out_dim: 9,
                units: vec![64, 64],
                activation_out: false,
            };
            BcModelConfig::default().policy_model_config(policy_model_config)
        };
        BcConfig::<Mlp>::default()
            .policy_model_config(policy_model_config)
            .batch_size(args.batch_size)
            .device(Device::Cpu)
            .action_type(BcActionType::Continuous)
    };
    let mut agent = Bc::build(agent_config.clone());

    // Create recorder
    let mut recorder: Box<dyn AggregateRecorder> = {
        match args.mlflow {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("D4RL")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&agent_config)?;
                recorder_run.set_tag("env", "kitchen")?;
                recorder_run.set_tag("algo", "bc")?;
                recorder_run.set_tag("backend", "candle")?;
                Box::new(recorder_run)
            }
            false => Box::new(TensorboardRecorder::new(MODEL_DIR)),
        }
    };

    // Create evaluator
    let mut evaluator = DefaultEvaluator::new(env, args.eval_episodes)?;

    // Start training
    let _ = trainer.train_offline(&mut agent, &mut buffer, &mut recorder, &mut evaluator);

    Ok(())
}
