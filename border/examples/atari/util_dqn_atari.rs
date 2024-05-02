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
            default = "default_flush_record_interval",
            skip_serializing_if = "is_default_flush_record_interval"
        )]
        pub flush_record_interval: usize,

        #[serde(
            default = "default_record_agent_info_interval",
            skip_serializing_if = "is_default_record_agent_info_interval"
        )]
        pub record_agent_info_interval: usize,

        #[serde(
            default = "default_record_compute_cost_interval",
            skip_serializing_if = "is_default_record_compute_cost_interval"
        )]
        pub record_compute_cost_interval: usize,

        #[serde(
            default = "default_warmup_period",
            skip_serializing_if = "is_default_warmup_period"
        )]
        pub warmup_period: usize,

        #[serde(
            default = "default_save_interval",
            skip_serializing_if = "is_default_save_interval"
        )]
        pub save_interval: usize,
    }

    impl Default for DqnAtariTrainerConfig {
        fn default() -> Self {
            Self {
                model_dir: "".to_string(),
                max_opts: 3000000,
                opt_interval: 1,
                eval_interval: 5000,
                record_agent_info_interval: 5000,
                record_compute_cost_interval: 5000,
                flush_record_interval: 5000,
                warmup_period: 2500,
                save_interval: 500000,
                // // For debug
                // model_dir: "".to_string(),
                // max_opts: 3000000,
                // opt_interval: 1,
                // eval_interval: 10,
                // record_agent_info_interval: 10,
                // record_compute_cost_interval: 10,
                // flush_record_interval: 10,
                // warmup_period: 32,
                // save_interval: 100,
            }
        }
    }

    fn default_max_opts() -> usize {
        DqnAtariTrainerConfig::default().max_opts
    }

    fn default_opt_interval() -> usize {
        DqnAtariTrainerConfig::default().opt_interval
    }

    fn default_eval_interval() -> usize {
        DqnAtariTrainerConfig::default().eval_interval
    }

    fn default_flush_record_interval() -> usize {
        DqnAtariTrainerConfig::default().flush_record_interval
    }

    fn default_record_agent_info_interval() -> usize {
        DqnAtariTrainerConfig::default().record_agent_info_interval
    }

    fn default_record_compute_cost_interval() -> usize {
        DqnAtariTrainerConfig::default().record_compute_cost_interval
    }

    fn default_warmup_period() -> usize {
        DqnAtariTrainerConfig::default().warmup_period
    }

    fn default_save_interval() -> usize {
        DqnAtariTrainerConfig::default().save_interval
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

    fn is_default_flush_record_interval(v: &usize) -> bool {
        *v == default_flush_record_interval()
    }

    fn is_default_record_agent_info_interval(v: &usize) -> bool {
        *v == default_record_agent_info_interval()
    }

    fn is_default_record_compute_cost_interval(v: &usize) -> bool {
        *v == default_record_compute_cost_interval()
    }

    fn is_default_warmup_period(v: &usize) -> bool {
        *v == default_warmup_period()
    }

    fn is_default_save_interval(v: &usize) -> bool {
        *v == default_save_interval()
    }

    impl Into<TrainerConfig> for DqnAtariTrainerConfig {
        fn into(self) -> TrainerConfig {
            TrainerConfig {
                model_dir: Some(self.model_dir),
                max_opts: self.max_opts,
                opt_interval: self.opt_interval,
                eval_interval: self.eval_interval,
                flush_record_interval: self.flush_record_interval,
                record_agent_info_interval: self.record_agent_info_interval,
                record_compute_cost_interval: self.record_compute_cost_interval,
                warmup_period: self.warmup_period,
                save_interval: self.save_interval,
            }
        }
    }
}

// #[cfg(feature = "border-async-trainer")]
mod async_trainer_config {
    use border_async_trainer::AsyncTrainerConfig;
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Serialize)]
    pub struct DqnAtariAsyncTrainerConfig {
        pub model_dir: Option<String>,

        #[serde(
            default = "default_max_opts",
            skip_serializing_if = "is_default_max_opts"
        )]
        pub max_opts: usize,

        #[serde(
            default = "default_eval_interval",
            skip_serializing_if = "is_default_eval_interval"
        )]
        pub eval_interval: usize,

        #[serde(
            default = "default_flush_record_interval",
            skip_serializing_if = "is_default_flush_record_interval"
        )]
        pub flush_record_interval: usize,

        #[serde(
            default = "default_record_compute_cost_interval",
            skip_serializing_if = "is_default_record_compute_cost_interval"
        )]
        pub record_compute_cost_interval: usize,

        #[serde(
            default = "default_save_interval",
            skip_serializing_if = "is_default_save_interval"
        )]
        pub save_interval: usize,

        #[serde(
            default = "default_sync_interval",
            skip_serializing_if = "is_default_sync_interval"
        )]
        pub sync_interval: usize,

        #[serde(
            default = "default_warmup_period",
            skip_serializing_if = "is_default_warmup_period"
        )]
        pub warmup_period: usize,
    }

    impl Default for DqnAtariAsyncTrainerConfig {
        fn default() -> Self {
            Self {
                model_dir: None,
                max_opts: 3000000,
                eval_interval: 5000,
                flush_record_interval: 5000,
                record_compute_cost_interval: 5000,
                sync_interval: 1,
                save_interval: 500000,
                warmup_period: 10000,
            }
        }
    }

    fn default_max_opts() -> usize {
        DqnAtariAsyncTrainerConfig::default().max_opts
    }

    fn default_eval_interval() -> usize {
        DqnAtariAsyncTrainerConfig::default().eval_interval
    }

    fn default_flush_record_interval() -> usize {
        DqnAtariAsyncTrainerConfig::default().flush_record_interval
    }

    fn default_record_compute_cost_interval() -> usize {
        DqnAtariAsyncTrainerConfig::default().record_compute_cost_interval
    }

    fn default_sync_interval() -> usize {
        DqnAtariAsyncTrainerConfig::default().sync_interval
    }

    fn default_save_interval() -> usize {
        DqnAtariAsyncTrainerConfig::default().save_interval
    }

    fn default_warmup_period() -> usize {
        DqnAtariAsyncTrainerConfig::default().warmup_period
    }

    fn is_default_max_opts(v: &usize) -> bool {
        *v == default_max_opts()
    }

    fn is_default_eval_interval(v: &usize) -> bool {
        *v == default_eval_interval()
    }

    fn is_default_flush_record_interval(v: &usize) -> bool {
        *v == default_flush_record_interval()
    }

    fn is_default_record_compute_cost_interval(v: &usize) -> bool {
        *v == default_record_compute_cost_interval()
    }

    fn is_default_sync_interval(v: &usize) -> bool {
        *v == default_sync_interval()
    }

    fn is_default_save_interval(v: &usize) -> bool {
        *v == default_save_interval()
    }

    fn is_default_warmup_period(v: &usize) -> bool {
        *v == default_warmup_period()
    }

    impl Into<AsyncTrainerConfig> for DqnAtariAsyncTrainerConfig {
        fn into(self) -> AsyncTrainerConfig {
            AsyncTrainerConfig {
                model_dir: self.model_dir,
                max_opts: self.max_opts,
                eval_interval: self.eval_interval,
                flush_record_interval: self.flush_record_interval,
                record_compute_cost_interval: self.record_compute_cost_interval,
                save_interval: self.save_interval,
                sync_interval: self.sync_interval,
                warmup_period: self.warmup_period,
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

    impl Default for DqnAtariReplayBufferConfig {
        fn default() -> Self {
            Self {
                capacity: 262144,
                seed: 42,
                per_config: None,
            }
        }
    }

    fn default_capacity() -> usize {
        DqnAtariReplayBufferConfig::default().capacity
    }

    fn default_seed() -> u64 {
        DqnAtariReplayBufferConfig::default().seed
    }

    fn default_per_config() -> Option<PerConfig> {
        DqnAtariReplayBufferConfig::default().per_config
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

#[cfg(feature = "tch")]
mod tch_dqn_config {
    use std::marker::PhantomData;

    use border_tch_agent::{
        cnn::{Cnn, CnnConfig},
        dqn::{DqnConfig, DqnExplorer, DqnModelConfig, EpsilonGreedy},
        opt::OptimizerConfig,
        util::CriticLoss,
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

        #[serde(
            default = "default_critic_loss",
            skip_serializing_if = "is_default_critic_loss"
        )]
        pub critic_loss: CriticLoss,

        #[serde(
            default = "default_record_verbose_level",
            skip_serializing_if = "is_default_record_verbose_level"
        )]
        pub record_verbose_level: usize,

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
                    opt_config: OptimizerConfig::Adam { lr: 0.0001 },
                },
                soft_update_interval: 10000,
                n_updates_per_opt: 1,
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
                critic_loss: CriticLoss::Mse,
                record_verbose_level: 0,
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

    fn default_critic_loss() -> CriticLoss {
        DqnAtariAgentConfig::default().critic_loss
    }

    fn default_record_verbose_level() -> usize {
        DqnAtariAgentConfig::default().record_verbose_level
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

    fn is_default_critic_loss(critic_loss: &CriticLoss) -> bool {
        critic_loss == &default_critic_loss()
    }

    fn is_default_record_verbose_level(record_verbose_level: &usize) -> bool {
        record_verbose_level == &default_record_verbose_level()
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
                batch_size: self.batch_size,
                discount_factor: self.discount_factor,
                tau: self.tau,
                train: self.train,
                explorer: self.explorer,
                clip_reward: self.clip_reward,
                double_dqn: self.double_dqn,
                clip_td_err: self.clip_td_err,
                device: self.device,
                critic_loss: self.critic_loss,
                record_verbose_level: self.record_verbose_level,
                phantom: PhantomData,
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
        util::CriticLoss,
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

        #[serde(
            default = "default_critic_loss",
            skip_serializing_if = "is_default_critic_loss"
        )]
        pub critic_loss: CriticLoss,

        #[serde(
            default = "default_record_verbose_level",
            skip_serializing_if = "is_default_record_verbose_level"
        )]
        pub record_verbose_level: usize,

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
                    opt_config: OptimizerConfig::Adam { lr: 0.0001 },
                },
                soft_update_interval: 10000,
                n_updates_per_opt: 1,
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
                critic_loss: CriticLoss::Mse,
                record_verbose_level: 0,
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

    fn default_critic_loss() -> CriticLoss {
        DqnAtariAgentConfig::default().critic_loss
    }

    fn default_record_verbose_level() -> usize {
        DqnAtariAgentConfig::default().record_verbose_level
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

    fn is_default_critic_loss(critic_loss: &CriticLoss) -> bool {
        critic_loss == &default_critic_loss()
    }

    fn is_default_record_verbose_level(record_verbose_level: &usize) -> bool {
        record_verbose_level == &default_record_verbose_level()
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
                batch_size: self.batch_size,
                discount_factor: self.discount_factor,
                tau: self.tau,
                train: self.train,
                explorer: self.explorer,
                clip_reward: self.clip_reward,
                double_dqn: self.double_dqn,
                clip_td_err: self.clip_td_err,
                device: self.device,
                critic_loss: self.critic_loss,
                record_verbose_level: self.record_verbose_level,
                phantom: PhantomData,
            }
        }
    }
}

#[allow(unused_imports)]
pub use replay_buffer_config::DqnAtariReplayBufferConfig;
#[allow(unused_imports)]
pub use trainer_config::DqnAtariTrainerConfig;

#[cfg(feature = "border-async-trainer")]
pub use async_trainer_config::DqnAtariAsyncTrainerConfig;

#[allow(unused_imports)]
#[cfg(feature = "candle-core")]
pub use candle_dqn_config::DqnAtariAgentConfig;
#[allow(unused_imports)]
#[cfg(feature = "tch")]
pub use tch_dqn_config::DqnAtariAgentConfig;
