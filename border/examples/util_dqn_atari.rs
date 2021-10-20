use anyhow::Result;
use border_tch_agent::dqn::{DQNExplorer, EpsilonGreedy};
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

    pub fn replay_buffer_capacity(mut self, replay_buffer_capacity: usize) -> Self {
        self.replay_buffer_capacity = replay_buffer_capacity;
        self
    }

    pub fn max_opts(mut self, max_opts: usize) -> Self {
        self.max_opts = max_opts;
        self
    }

    pub fn optimizer(mut self, optimizer: &'a str) -> Self {
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
