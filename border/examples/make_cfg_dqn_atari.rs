mod util_dqn_atari;
use anyhow::Result;
use border_core::{replay_buffer::SimpleReplayBufferConfig, TrainerConfig};
use border_tch_agent::{
    cnn::{CNNConfig, CNN},
    dqn::{DQNConfig, DQNModelConfig}, //, EpsilonGreedy, DQNExplorer},
    opt::OptimizerConfig,
};
use std::{default::Default, path::Path};
use util_dqn_atari::{Params, model_dir};

fn make_dqn_config(params: &Params) -> DQNConfig<CNN> {
    let n_stack = 4;
    let out_dim = 0; // Set before training/evaluation
    let model_config = DQNModelConfig::default()
        .q_config(CNNConfig::new(n_stack, out_dim))
        .out_dim(out_dim)
        .opt_config(OptimizerConfig::Adam { lr: params.lr });
    DQNConfig::default()
        .model_config(model_config)
        .batch_size(params.batch_size)
        .discount_factor(params.discount_factor)
        .double_dqn(params.double_dqn)
        .min_transitions_warmup(params.min_transition_warmup)
        .soft_update_interval(params.soft_update_interval)
        .clip_reward(params.clip_reward)
        .tau(params.tau)
        .explorer(params.explorer.clone())
}

fn make_replay_buffer_config(params: &Params) -> SimpleReplayBufferConfig {
    SimpleReplayBufferConfig::default().capacity(params.replay_buffer_capacity)
}

fn make_trainer_config(env_name: String, params: &Params) -> Result<TrainerConfig> {
    let model_dir = model_dir(env_name, params)?;

    let (max_opts, opt_interval, record_interval) = if !params.debug {
        (params.max_opts, params.opt_interval, params.record_interval)
    } else {
        (1000, 100, 100)
    };

    Ok(TrainerConfig::default()
        .max_opts(max_opts)
        .opt_interval(opt_interval)
        .record_interval(record_interval)
        .eval_interval(params.eval_interval)
        .eval_episodes(params.eval_episodes)
        .model_dir(model_dir)
        .save_interval(params.save_interval))
}

fn make_cfg(env_name: impl Into<String> + Clone, params: &Params) -> Result<()> {
    let agent_config = make_dqn_config(params);
    let replay_buffer_config = make_replay_buffer_config(params);
    let trainer_config = make_trainer_config(env_name.clone().into(), params)?;

    let model_dir = model_dir(env_name.into(), params)?;
    let agent_path = Path::new(&model_dir).join("agent.yaml");
    let replay_buffer_path = Path::new(&model_dir).join("replay_buffer.yaml");
    let trainer_path = Path::new(&model_dir).join("trainer.yaml");

    agent_config.save(&agent_path)?;
    replay_buffer_config.save(&replay_buffer_path)?;
    trainer_config.save(&trainer_path)?;

    println!("{:?}", agent_path);
    println!("{:?}", replay_buffer_path);
    println!("{:?}", trainer_path);

    Ok(())
}

fn main() -> Result<()> {
    // Pong
    let params = Params::default().replay_buffer_capacity(65536);
    make_cfg("PongNoFrameskip-v4", &params)?;
    make_cfg("PongNoFrameskip-v4", &params.clone().debug())?;
    // make_cfg("PongNoFrameskip-v4", &params.clone().per())?;
    // make_cfg("PongNoFrameskip-v4", &params.clone().ddqn())?;

    // Hero
    let params = Params::default()
        .max_opts(50_000_000)
        .eval_interval(500_000)
        .save_interval(10_000_000)
        .replay_buffer_capacity(1048576)
        .optimizer("adam");
    make_cfg("HeroNoFrameskip-v4", &params)?;
    // make_cfg("HeroNoFrameskip-v4", &params.clone().per())?;
    // make_cfg("HeroNoFrameskip-v4", &params.clone().ddqn())?;

    Ok(())
}
