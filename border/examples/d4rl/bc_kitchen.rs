use anyhow::Result;
use border_candle_agent::{
    bc::{Bc, BcActionType, BcConfig, BcModelConfig},
    mlp::{Mlp, MlpConfig},
};
use border_core::{
    Agent, Configurable, DefaultEvaluator, Env, ExperienceBufferBase, Trainer, TrainerConfig,
    record::AggregateRecorder,
};
use border_minari::{d4rl::kitchen::candle::KitchenConverter, MinariDataset, MinariEnv};
use border_mlflow_tracking::MlflowTrackingClient;
use candle_core::Device;
use core::panic;
use numpy::convert;
use std::num;

fn main() -> Result<()> {
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
    let mut env = dataset.recover_environment(converter, false, None)?;

    // Create trainer
    let mut trainer = Trainer::build(TrainerConfig::default());

    // Create behavior cloning agent
    let agent_config = {
        let policy_model_config = {
            let policy_model_config = MlpConfig {
                in_dim: 14,
                out_dim: 9,
                units: vec![64, 64],
                activation_out: false,
            };
            BcModelConfig::default().policy_model_config(policy_model_config)
        };
        BcConfig::<Mlp>::default()
            .policy_model_config(policy_model_config)
            .device(Device::Cpu)
            .action_type(BcActionType::Continuous)
    };
    let mut agent = Bc::build(agent_config.clone());

    // Create recorder
    let mut recorder: Box<dyn AggregateRecorder> = {
        let client =
            MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("D4RL")?;
        let recorder_run = client.create_recorder("")?;
        recorder_run.log_params(&agent_config)?;
        recorder_run.set_tag("env", "kitchen")?;
        recorder_run.set_tag("algo", "bc")?;
        recorder_run.set_tag("backend", "candle")?;
        Box::new(recorder_run)
    };

    // Create evaluator
    let mut evaluator = DefaultEvaluator::new(env, 5)?;

    // Start training
    trainer.train_offline(&mut agent, &mut buffer, &mut recorder, &mut evaluator);

    Ok(())
}
