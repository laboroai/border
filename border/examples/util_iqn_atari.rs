use anyhow::Result;
use border_tch_agent::{iqn::{IqnExplorer, EpsilonGreedy}, opt::OptimizerConfig};
use std::{default::Default, path::Path};

#[derive(Clone)]
pub struct Params {
    // Model (hyper)parameters
    pub feature_dim: i64,
    pub hidden_dim: i64,
    pub embed_dim: i64,
    pub opt_config: OptimizerConfig,

    // Agent parameters
    pub replay_buffer_capacity: usize,
    pub per: bool,
    pub batch_size: usize,
    pub discount_factor: f64,
    pub min_transition_warmup: usize,
    pub soft_update_interval: usize,
    pub explorer: IqnExplorer,
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

impl Default for Params {
    fn default() -> Self {
        Self {
            // Model (hyper)parameters
            feature_dim: 3136,
            hidden_dim: 512,
            embed_dim: 64,
            opt_config: OptimizerConfig::Adam {
                lr: 1e-5,
                // eps: 0.01 / 32.0
            },
        
            // Agent parameters
            replay_buffer_capacity: 1_048_576,
            per: false,
            batch_size: 32,
            discount_factor: 0.99,
            min_transition_warmup: 2500,
            soft_update_interval: 10_000,
            tau: 1.0,
            explorer: EpsilonGreedy::with_params(
                1.0,
                0.02,
                1_000_000,
            ),

            // Trainer parameters
            max_opts: 50_000_000,
            eval_interval: 500_000,
            eval_episodes: 1,
            opt_interval: 1,
            record_interval: 500_000,
            save_interval: 10_000_000,

            // Debug parameters
            debug: false,
        }
    }
}

impl Params {
    #[allow(dead_code)]
    pub fn per(mut self) -> Self {
        self.per = true;
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
}

#[allow(dead_code)]
pub fn model_dir(env_name: String, params: &Params) -> Result<String> {
    let per = params.per;
    // let debug = params.debug;

    let mut model_dir = format!("./border/examples/model/iqn_{}", env_name);

    if per {
        model_dir.push_str("_per");
    }

    // if debug {
    //     model_dir.push_str("_debug");
    // }

    if !Path::new(&model_dir).exists() {
        std::fs::create_dir(Path::new(&model_dir))?;
    }

    Ok(model_dir)
}

#[allow(dead_code)]
pub fn model_dir_async(env_name: String, params: &Params) -> Result<String> {
    let per = params.per;
    // let debug = params.debug;

    let mut model_dir = format!("./border/examples/model/iqn_{}", env_name);

    // if ddqn {
    //     model_dir.push_str("_ddqn");
    // }

    if per {
        model_dir.push_str("_per");
    }

    // if debug {
    //     model_dir.push_str("_debug");
    // }

    model_dir.push_str("_async");

    if !Path::new(&model_dir).exists() {
        std::fs::create_dir(Path::new(&model_dir))?;
    }

    Ok(model_dir)
}
