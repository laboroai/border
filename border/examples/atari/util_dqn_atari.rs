use anyhow::Result;
use border_tch_agent::dqn::{DqnExplorer, EpsilonGreedy};
use std::{default::Default, path::Path};

mod trainer_config {
    use border_core::TrainerConfig;
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Serialize)]
    pub struct DqnAtariTrainerConfig {
        pub model_dir: String,

        #[serde(
            default = "default_max_opts",
            skip_serializing_if = "is_default_max_opts"
        )]
        pub max_opts: usize,

        #[serde(
            default = "default_opt_interval",
            skip_serializing_if = "is_default_opt_interval"
        )]
        pub opt_interval: usize,

        #[serde(
            default = "default_eval_interval",
            skip_serializing_if = "is_default_eval_interval"
        )]
        pub eval_interval: usize,

        #[serde(
            default = "default_record_interval",
            skip_serializing_if = "is_default_record_interval"
        )]
        pub record_interval: usize,

        #[serde(
            default = "default_save_interval",
            skip_serializing_if = "is_default_save_interval"
        )]
        pub save_interval: usize,
    }

    fn default_max_opts() -> usize {
        3000000
    }

    fn default_opt_interval() -> usize {
        1
    }

    fn default_eval_interval() -> usize {
        50000
    }

    fn default_record_interval() -> usize {
        50000
    }

    fn default_save_interval() -> usize {
        500000
    }

    fn is_default_max_opts(v: &usize) -> bool {
        *v == default_max_opts()
    }

    fn is_default_opt_interval(v: &usize) -> bool {
        *v == default_opt_interval()
    }

    fn is_default_eval_interval(v: &usize) -> bool {
        *v == default_eval_interval()
    }

    fn is_default_record_interval(v: &usize) -> bool {
        *v == default_record_interval()
    }

    fn is_default_save_interval(v: &usize) -> bool {
        *v == default_save_interval()
    }

    impl Default for DqnAtariTrainerConfig {
        fn default() -> Self {
            Self {
                model_dir: "".to_string(),
                max_opts: default_max_opts(),
                opt_interval: default_opt_interval(),
                eval_interval: default_eval_interval(),
                record_interval: default_record_interval(),
                save_interval: default_save_interval(),
            }
        }
    }

    impl Into<TrainerConfig> for DqnAtariTrainerConfig {
        fn into(self) -> TrainerConfig {
            TrainerConfig {
                model_dir: Some(self.model_dir),
                max_opts: self.max_opts,
                opt_interval: self.opt_interval,
                eval_interval: self.eval_interval,
                record_interval: self.record_interval,
                save_interval: self.save_interval,
            }
        }
    }
}

pub use trainer_config::DqnAtariTrainerConfig;

#[derive(Clone)]
pub struct Params<'a> {
    // Agent parameters
    pub replay_buffer_capacity: usize,
    pub per: bool,
    pub double_dqn: bool,
    pub optimizer: &'a str,
    pub batch_size: usize,
    pub discount_factor: f64,
    pub min_transition_warmup: usize,
    pub soft_update_interval: usize,
    pub lr: f64,
    pub clip_reward: Option<f64>,
    pub explorer: DqnExplorer,
    pub tau: f64,

    // Trainer parameters
    pub max_opts: usize,
    pub eval_interval: usize,
    pub eval_episodes: usize,
    pub opt_interval: usize,
    pub record_interval: usize,
    pub save_interval: usize,

    // Debug parameters
    pub debug: bool,
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
            tau: 1.0,

            // Trainer parameters
            max_opts: 3_000_000,
            eval_interval: 50_000,
            eval_episodes: 1,
            opt_interval: 1,
            record_interval: 50_000,
            save_interval: 500_000,

            // Debug parameters
            debug: false,
        }
    }
}

impl<'a> Params<'a> {
    #[allow(dead_code)]
    pub fn per(mut self) -> Self {
        self.per = true;
        self
    }

    #[allow(dead_code)]
    pub fn ddqn(mut self) -> Self {
        self.double_dqn = true;
        self
    }

    #[allow(dead_code)]
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    #[allow(dead_code)]
    pub fn replay_buffer_capacity(mut self, replay_buffer_capacity: usize) -> Self {
        self.replay_buffer_capacity = replay_buffer_capacity;
        self
    }

    #[allow(dead_code)]
    pub fn max_opts(mut self, max_opts: usize) -> Self {
        self.max_opts = max_opts;
        self
    }

    #[allow(dead_code)]
    pub fn save_interval(mut self, save_interval: usize) -> Self {
        self.save_interval = save_interval;
        self
    }

    #[allow(dead_code)]
    pub fn eval_interval(mut self, eval_interval: usize) -> Self {
        self.eval_interval = eval_interval;
        self
    }

    #[allow(dead_code)]
    pub fn optimizer(mut self, optimizer: &'a str) -> Self {
        self.optimizer = optimizer;
        self
    }
}

#[allow(dead_code)]
pub fn model_dir_async(env_name: String, params: &Params) -> Result<String> {
    let per = params.per;
    let ddqn = params.double_dqn;
    let debug = params.debug;

    let mut model_dir = format!("./border/examples/atari/model/dqn_{}", env_name);
    if ddqn {
        model_dir.push_str("_ddqn");
    }

    if per {
        model_dir.push_str("_per");
    }

    if debug {
        model_dir.push_str("_debug");
    }

    model_dir.push_str("_async");

    if !Path::new(&model_dir).exists() {
        std::fs::create_dir(Path::new(&model_dir))?;
    }

    Ok(model_dir)
}
