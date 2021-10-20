use anyhow::Result;
use border_core::{replay_buffer::SimpleReplayBufferConfig, TrainerConfig};
use border_tch_agent::{
    cnn::{CNNConfig, CNN},
    dqn::{DQNConfig, DQNModelConfig, EpsilonGreedy, DQNExplorer},
    opt::OptimizerConfig,
};
use std::{default::Default, path::Path};

#[derive(Clone)]
pub struct Params<'a> {
    // Agent parameters
    replay_buffer_capacity: usize,
    per: bool,
    double_dqn: bool,
    optimizer: &'a str,
    batch_size: usize,
    discount_factor: f64,
    min_transition_warmup: usize,
    soft_update_interval: usize,
    lr: f64,
    clip_reward: Option<f64>,
    explorer: DQNExplorer,

    // Trainer parameters
    max_opts: usize,
    eval_interval: usize,
    eval_episodes: usize,
    opt_interval: usize,
    record_interval: usize,
    save_interval: usize,
}

impl<'a> Default for Params<'a> {
    fn default() -> Self {
        Self {
            // Agent parameters
            replay_buffer_capacity: 50_000,
            per: false,
            double_dqn: false,
            optimizer: "adam",
            batch_size: 32,
            discount_factor: 0.99,
            min_transition_warmup: 2500,
            soft_update_interval: 10_000,
            lr: 1e-4,
            clip_reward: Some(1.0),
            explorer: EpsilonGreedy::with_final_step(1_000_000),

            // Trainer parameters
            max_opts: 3_000_000,
            eval_interval: 50_000,
            eval_episodes: 1,
            opt_interval: 1,
            record_interval: 50_000,
            save_interval: 500_000,
        }
    }
}

impl<'a> Params<'a> {
    pub fn per(mut self) -> Self {
        self.per = true;
        self
    }

    pub fn ddqn(mut self) -> Self {
        self.double_dqn = true;
        self
    }

    fn replay_buffer_capacity(mut self, replay_buffer_capacity: usize) -> Self {
        self.replay_buffer_capacity = replay_buffer_capacity;
        self
    }

    fn max_opts(mut self, max_opts: usize) -> Self {
        self.max_opts = max_opts;
        self
    }

    fn optimizer(mut self, optimizer: &'a str) -> Self {
        self.optimizer = optimizer;
        self
    }
}

pub fn model_dir(env_name: String, params: &Params) -> Result<String> {
    let per = params.per;
    let ddqn = params.double_dqn;

    let mut model_dir = format!("./border/examples/model/dqn_{}", env_name);
    if per {
        model_dir.push_str("_per");
    }

    if ddqn {
        model_dir.push_str("_ddqn");
    }

    if !Path::new(&model_dir).exists() {
        std::fs::create_dir(Path::new(&model_dir))?;
    }

    Ok(model_dir)
}

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
        .explorer(params.explorer.clone())
}

fn make_replay_buffer_config(params: &Params) -> SimpleReplayBufferConfig {
    SimpleReplayBufferConfig::default().capacity(params.replay_buffer_capacity)
}

fn make_trainer_config(env_name: String, params: &Params) -> Result<TrainerConfig> {
    let model_dir = model_dir(env_name, params)?;

    Ok(TrainerConfig::default()
        .max_opts(params.max_opts)
        .eval_interval(params.eval_interval)
        .eval_episodes(params.eval_episodes)
        .model_dir(model_dir)
        .opt_interval(params.opt_interval)
        .record_interval(params.record_interval)
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
    make_cfg("PongNoFrameskip-v4", &params.clone().per())?;
    make_cfg("PongNoFrameskip-v4", &params.clone().ddqn())?;

    // Hero
    let params = Params::default()
        .max_opts(50_000_000)
        .replay_buffer_capacity(1048576)
        .optimizer("adam");
    make_cfg("HeroNoFrameskip-v4", &params)?;
    make_cfg("HeroNoFrameskip-v4", &params.clone().per())?;
    make_cfg("HeroNoFrameskip-v4", &params.clone().ddqn())?;

    Ok(())
}
