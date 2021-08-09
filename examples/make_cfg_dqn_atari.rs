mod dqn_atari_model;
use dqn_atari_model::{CNNConfig, CNN};
use anyhow::Result;
use border_core::TrainerBuilder;
use border_tch_agent::{
    dqn::{EpsilonGreedy, DQNBuilder, DQNModelBuilder},
    opt::OptimizerConfig,
    util::OptInterval,
    replay_buffer::{ExperienceSampling, IwScheduler}
};
use std::{default::Default, path::Path};

// DQN agent parameters
const N_STACK: i64 = 4;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 32;
const N_TRANSITIONS_WARMUP: usize = 2500;
const N_UPDATES_PER_OPT: usize = 1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const SOFT_UPDATE_INTERVAL: usize = 10_000;
const TAU: f64 = 1.0;
const EPS_FINAL_STEP: usize = 1_000_000;
const REPLAY_BUFFER_CAPACITY: usize = 50_000;

// DQN model parameters
const LR_QNET: f64 = 1e-4;

// Training parameters
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const N_EPISODES_PER_EVAL: usize = 1;

fn saving_model_dir(env_name: String, per: bool) -> Result<String> {
    let mut model_dir = format!("./examples/model/dqn_{}", env_name);
    if per {
        model_dir.push_str("_per");
    }

    if !Path::new(&model_dir).exists() {
        std::fs::create_dir(Path::new(&model_dir))?;
    }

    Ok(model_dir)
}

fn make_cfg(env_name: impl Into<String>, per: bool) -> Result<()> {
    let saving_model_dir = saving_model_dir(env_name.into(), per)?;
    let model_cfg = Path::new(&saving_model_dir).join("model.yaml");
    let agent_cfg = Path::new(&saving_model_dir).join("agent.yaml");
    let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
    println!("{:?}", agent_cfg);

    let out_dim = 0; // set in training/evaluation code
    let builder = DQNModelBuilder::<CNN>::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_QNET })
        .q_config(CNNConfig::new(N_STACK, out_dim));
    let _ = builder.save(model_cfg);

    let builder = DQNBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .soft_update_interval(SOFT_UPDATE_INTERVAL)
        .tau(TAU)
        .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY)
        .explorer(EpsilonGreedy::with_final_step(EPS_FINAL_STEP));
    let builder = if per {
        builder
            .expr_sampling(
                ExperienceSampling::TDerror {
                    alpha: 0.7f32,
                    iw_scheduler: IwScheduler {
                        beta_0: 0.5f32,
                        beta_final: 1f32,
                        n_opts_final: 1_000_000,
                    }
                }
            )
    } else {
        builder
    };
    let _ = builder.save(agent_cfg);

    let builder = TrainerBuilder::default()
        .max_opts(MAX_OPTS)
        .eval_interval(EVAL_INTERVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL)
        .model_dir(saving_model_dir);
    let _ = builder.save(trainer_cfg);
    Ok(())
}

fn main() -> Result<()> {
    make_cfg("PongNoFrameskip-v4", false)?;
    make_cfg("PongNoFrameskip-v4", true)?;

    Ok(())
}
