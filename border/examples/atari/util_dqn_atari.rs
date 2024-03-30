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

mod replay_buffer_config {
    use border_core::replay_buffer::{PerConfig, SimpleReplayBufferConfig};
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Serialize)]
    pub struct DqnAtariReplayBufferConfig {
        #[serde(
            default = "default_capacity",
            skip_serializing_if = "is_default_capacity"
        )]
        pub capacity: usize,

        #[serde(default = "default_seed", skip_serializing_if = "is_default_seed")]
        pub seed: u64,

        /// Currently, fixed to None
        #[serde(
            default = "default_per_config",
            skip_serializing_if = "is_default_per_config"
        )]
        pub per_config: Option<PerConfig>,
    }

    fn default_capacity() -> usize {
        262144
    }

    fn default_seed() -> u64 {
        42
    }

    fn default_per_config() -> Option<PerConfig> {
        None
    }

    fn is_default_capacity(v: &usize) -> bool {
        *v == default_capacity()
    }

    fn is_default_seed(v: &u64) -> bool {
        *v == default_seed()
    }

    fn is_default_per_config(v: &Option<PerConfig>) -> bool {
        *v == default_per_config()
    }

    impl Default for DqnAtariReplayBufferConfig {
        fn default() -> Self {
            Self {
                capacity: default_capacity(),
                seed: default_seed(),
                per_config: default_per_config(),
            }
        }
    }

    impl Into<SimpleReplayBufferConfig> for DqnAtariReplayBufferConfig {
        fn into(self) -> SimpleReplayBufferConfig {
            SimpleReplayBufferConfig {
                capacity: self.capacity,
                seed: self.seed,
                per_config: self.per_config,
            }
        }
    }
}

#[cfg(feature = "candle-core")]
mod candle_dqn_config {
    use std::marker::PhantomData;

    use border_candle_agent::{
        cnn::{Cnn, CnnConfig},
        dqn::{DqnConfig, DqnExplorer, DqnModelConfig, EpsilonGreedy},
        opt::OptimizerConfig,
        Device,
    };
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Serialize)]
    pub struct DqnAtariAgentConfig {
        #[serde(
            default = "default_model_config",
            skip_serializing_if = "is_default_model_config"
        )]
        pub model_config: DqnModelConfig<CnnConfig>,
        #[serde(
            default = "default_soft_update_interval",
            skip_serializing_if = "is_default_soft_update_interval"
        )]
        pub soft_update_interval: usize,
        #[serde(
            default = "default_n_updates_per_opt",
            skip_serializing_if = "is_default_n_updates_per_opt"
        )]
        pub n_updates_per_opt: usize,
        #[serde(
            default = "default_min_transitions_warmup",
            skip_serializing_if = "is_default_min_transitions_warmup"
        )]
        pub min_transitions_warmup: usize,
        #[serde(
            default = "default_batch_size",
            skip_serializing_if = "is_default_batch_size"
        )]
        pub batch_size: usize,
        #[serde(
            default = "default_discount_factor",
            skip_serializing_if = "is_default_discount_factor"
        )]
        pub discount_factor: f64,
        #[serde(default = "default_tau", skip_serializing_if = "is_default_tau")]
        pub tau: f64,
        #[serde(default = "default_train", skip_serializing_if = "is_default_train")]
        pub train: bool,
        #[serde(
            default = "default_explorer",
            skip_serializing_if = "is_default_explorer"
        )]
        pub explorer: DqnExplorer,
        #[serde(
            default = "default_clip_reward",
            skip_serializing_if = "is_default_clip_reward"
        )]
        pub clip_reward: Option<f64>,
        #[serde(
            default = "default_double_dqn",
            skip_serializing_if = "is_default_double_dqn"
        )]
        pub double_dqn: bool,
        #[serde(
            default = "default_clip_td_err",
            skip_serializing_if = "is_default_clip_td_err"
        )]
        pub clip_td_err: Option<(f64, f64)>,
        #[serde(default = "default_device", skip_serializing_if = "is_default_device")]
        pub device: Option<Device>,
        // phantom: PhantomData<CnnConfig>,
    }

    impl Default for DqnAtariAgentConfig {
        fn default() -> Self {
            DqnAtariAgentConfig {
                model_config: DqnModelConfig {
                    q_config: Some(CnnConfig {
                        n_stack: 4,
                        out_dim: 0,
                        skip_linear: false,
                    }),
                    opt_config: OptimizerConfig::default(),
                },
                soft_update_interval: 10000,
                n_updates_per_opt: 1,
                min_transitions_warmup: 2500,
                batch_size: 32,
                discount_factor: 0.99,
                tau: 1.0,
                train: false,
                explorer: DqnExplorer::EpsilonGreedy(EpsilonGreedy {
                    n_opts: 0,
                    eps_start: 1.0,
                    eps_final: 0.02,
                    final_step: 1000000,
                }),
                clip_reward: Some(1.0),
                double_dqn: false,
                clip_td_err: None,
                device: None,
                // phantom: PhantomData,
            }
        }
    }

    fn default_model_config() -> DqnModelConfig<CnnConfig> {
        DqnAtariAgentConfig::default().model_config
    }

    fn default_soft_update_interval() -> usize {
        DqnAtariAgentConfig::default().soft_update_interval
    }

    fn default_n_updates_per_opt() -> usize {
        DqnAtariAgentConfig::default().n_updates_per_opt
    }

    fn default_min_transitions_warmup() -> usize {
        DqnAtariAgentConfig::default().min_transitions_warmup
    }

    fn default_batch_size() -> usize {
        DqnAtariAgentConfig::default().batch_size
    }

    fn default_discount_factor() -> f64 {
        DqnAtariAgentConfig::default().discount_factor
    }

    fn default_tau() -> f64 {
        DqnAtariAgentConfig::default().tau
    }

    fn default_train() -> bool {
        DqnAtariAgentConfig::default().train
    }

    fn default_explorer() -> DqnExplorer {
        DqnAtariAgentConfig::default().explorer
    }

    fn default_clip_reward() -> Option<f64> {
        DqnAtariAgentConfig::default().clip_reward
    }

    fn default_double_dqn() -> bool {
        DqnAtariAgentConfig::default().double_dqn
    }

    fn default_clip_td_err() -> Option<(f64, f64)> {
        DqnAtariAgentConfig::default().clip_td_err
    }

    fn default_device() -> Option<Device> {
        DqnAtariAgentConfig::default().device
    }

    fn is_default_model_config(config: &DqnModelConfig<CnnConfig>) -> bool {
        config == &default_model_config()
    }

    fn is_default_soft_update_interval(soft_update_interval: &usize) -> bool {
        soft_update_interval == &default_soft_update_interval()
    }

    fn is_default_n_updates_per_opt(n_updates_per_opt: &usize) -> bool {
        n_updates_per_opt == &default_n_updates_per_opt()
    }

    fn is_default_min_transitions_warmup(min_transitions_warmup: &usize) -> bool {
        min_transitions_warmup == &default_min_transitions_warmup()
    }

    fn is_default_batch_size(batch_size: &usize) -> bool {
        batch_size == &default_batch_size()
    }

    fn is_default_discount_factor(discount_factor: &f64) -> bool {
        discount_factor == &default_discount_factor()
    }

    fn is_default_tau(tau: &f64) -> bool {
        tau == &default_tau()
    }

    fn is_default_train(train: &bool) -> bool {
        train == &default_train()
    }

    fn is_default_explorer(explorer: &DqnExplorer) -> bool {
        explorer == &default_explorer()
    }

    fn is_default_clip_reward(clip_reward: &Option<f64>) -> bool {
        clip_reward == &default_clip_reward()
    }

    fn is_default_double_dqn(double_dqn: &bool) -> bool {
        double_dqn == &default_double_dqn()
    }

    fn is_default_clip_td_err(clip_td_err: &Option<(f64, f64)>) -> bool {
        clip_td_err == &default_clip_td_err()
    }

    fn is_default_device(device: &Option<Device>) -> bool {
        device == &default_device()
    }

    impl Into<DqnConfig<Cnn>> for DqnAtariAgentConfig {
        fn into(self) -> DqnConfig<Cnn> {
            DqnConfig {
                model_config: self.model_config,
                soft_update_interval: self.soft_update_interval,
                n_updates_per_opt: self.n_updates_per_opt,
                min_transitions_warmup: self.min_transitions_warmup,
                batch_size: self.batch_size,
                discount_factor: self.discount_factor,
                tau: self.tau,
                train: self.train,
                explorer: self.explorer,
                clip_reward: self.clip_reward,
                double_dqn: self.double_dqn,
                clip_td_err: self.clip_td_err,
                device: self.device,
                phantom: PhantomData,
            }
        }
    }
}

pub use replay_buffer_config::DqnAtariReplayBufferConfig;
pub use trainer_config::DqnAtariTrainerConfig;
#[cfg(feature = "candle-core")]
pub use candle_dqn_config::DqnAtariAgentConfig;

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
