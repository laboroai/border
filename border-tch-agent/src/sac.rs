//! SAC agent.
//!
//! Here is an example in `border/examples/sac_pendulum.rs`
//!
//! ```rust,ignore
//! fn create_agent(in_dim: i64, out_dim: i64) -> Sac<Env, Mlp, Mlp2, ReplayBuffer> {
//!     let device = tch::Device::cuda_if_available();
//!     let actor_config = ActorConfig::default()
//!         .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
//!         .out_dim(out_dim)
//!         .pi_config(MlpConfig::new(in_dim, vec![64, 64], out_dim, true));
//!     let critic_config = CriticConfig::default()
//!         .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
//!         .q_config(MlpConfig::new(in_dim + out_dim, vec![64, 64], 1, true));
//!     let sac_config = SacConfig::default()
//!         .batch_size(BATCH_SIZE)
//!         .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//!         .actor_config(actor_config)
//!         .critic_config(critic_config)
//!         .device(device);
//!     Sac::build(sac_config)
//! }
//!
//! fn train(max_opts: usize, model_dir: &str, eval_interval: usize) -> Result<()> {
//!     let trainer = //...
//!     let mut agent = create_agent(DIM_OBS, DIM_ACT);
//!     let mut recorder = TensorboardRecorder::new(model_dir);
//!     let mut evaluator = Evaluator::new(&env_config(), 0, N_EPISODES_PER_EVAL)?;
//!
//!     trainer.train(&mut agent, &mut recorder, &mut evaluator)?;
//!
//!     Ok(())
//! }
//! ```
mod actor;
mod base;
mod config;
mod critic;
mod ent_coef;
pub use actor::{Actor, ActorConfig};
pub use base::Sac;
pub use config::SacConfig;
pub use critic::{Critic, CriticConfig};
pub use ent_coef::{EntCoef, EntCoefMode};
