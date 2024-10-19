//! SAC agent.
//!
//! Here is an example of creating SAC agent:
//!
//! ```no_run
//! # use anyhow::Result;
//! use border_core::{
//! #     Env as Env_, Obs as Obs_, Act as Act_, Step, test::{
//! #         TestAct as TestAct_, TestActBatch as TestActBatch_,
//! #         TestEnv as TestEnv_,
//! #         TestObs as TestObs_, TestObsBatch as TestObsBatch_,
//! #     },
//! #     record::Record,
//! #     generic_replay_buffer::{SimpleReplayBuffer, BatchBase},
//!       Configurable,
//! };
//! use border_candle_agent::{
//!     sac::{ActorConfig, CriticConfig, Sac, SacConfig},
//!     mlp::{Mlp, Mlp2, MlpConfig},
//!     opt::OptimizerConfig
//! };
//!
//! # struct TestEnv(TestEnv_);
//! # #[derive(Clone, Debug)]
//! # struct TestObs(TestObs_);
//! # #[derive(Clone, Debug)]
//! # struct TestAct(TestAct_);
//! # struct TestObsBatch(TestObsBatch_);
//! # struct TestActBatch(TestActBatch_);
//! #
//! # impl Obs_ for TestObs {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl Into<candle_core::Tensor> for TestObs {
//! #     fn into(self) -> candle_core::Tensor {
//! #         unimplemented!();
//! #     }
//! # }
//! #
//! # impl BatchBase for TestObsBatch {
//! #     fn new(n: usize) -> Self {
//! #         Self(TestObsBatch_::new(n))
//! #     }
//! #
//! #     fn push(&mut self, ix: usize, data: Self) {
//! #         self.0.push(ix, data.0);
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         Self(self.0.sample(ixs))
//! #     }
//! # }
//! #
//! # impl BatchBase for TestActBatch {
//! #     fn new(n: usize) -> Self {
//! #         Self(TestActBatch_::new(n))
//! #     }
//! #
//! #     fn push(&mut self, ix: usize, data: Self) {
//! #         self.0.push(ix, data.0);
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         Self(self.0.sample(ixs))
//! #     }
//! # }
//! #
//! # impl Act_ for TestAct {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl From<candle_core::Tensor> for TestAct {
//! #     fn from(t: candle_core::Tensor) -> Self {
//! #         unimplemented!();
//! #     }
//! # }
//! #
//! # impl Into<candle_core::Tensor> for TestAct {
//! #     fn into(self) -> candle_core::Tensor {
//! #         unimplemented!();
//! #     }
//! # }
//! #
//! # impl Env_ for TestEnv {
//! #     type Config = <TestEnv_ as Env_>::Config;
//! #     type Obs = TestObs;
//! #     type Act = TestAct;
//! #     type Info = <TestEnv_ as Env_>::Info;
//! #
//! #     fn build(config: &Self::Config, seed: i64) -> Result<Self> {
//! #         Ok(Self(TestEnv_::build(&config, seed).unwrap()))
//! #     }
//! #
//! #     fn step(&mut self, act: &TestAct) -> (Step<Self>, Record) {
//! #         let (step, record) = self.0.step(&act.0);
//! #         let step = Step {
//! #             obs: TestObs(step.obs),
//! #             act: TestAct(step.act),
//! #             reward: step.reward,
//! #             is_terminated: step.is_terminated,
//! #             is_truncated: step.is_truncated,
//! #             info: step.info,
//! #             init_obs: Some(TestObs(step.init_obs.unwrap())),
//! #         };
//! #         (step, record)
//! #     }
//! #
//! #     fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<TestObs> {
//! #         Ok(TestObs(self.0.reset(is_done).unwrap()))
//! #     }
//! #
//! #     fn step_with_reset(&mut self, a: &TestAct) -> (Step<Self>, Record) {
//! #         let (step, record) = self.0.step_with_reset(&a.0);
//! #         let step = Step {
//! #             obs: TestObs(step.obs),
//! #             act: TestAct(step.act),
//! #             reward: step.reward,
//! #             is_terminated: step.is_terminated,
//! #             is_truncated: step.is_truncated,
//! #             info: step.info,
//! #             init_obs: Some(TestObs(step.init_obs.unwrap())),
//! #         };
//! #         (step, record)
//! #     }
//! #
//! #     fn reset_with_index(&mut self, ix: usize) -> Result<TestObs> {
//! #         Ok(TestObs(self.0.reset_with_index(ix).unwrap()))
//! #     }
//! # }
//! #
//! # type Env = TestEnv;
//! # type ObsBatch = TestObsBatch;
//! # type ActBatch = TestActBatch;
//! # type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! #
//! const DIM_OBS: i64 = 3;
//! const DIM_ACT: i64 = 1;
//! const LR_ACTOR: f64 = 1e-3;
//! const LR_CRITIC: f64 = 1e-3;
//! const BATCH_SIZE: usize = 256;
//!
//! fn create_agent(in_dim: i64, out_dim: i64) -> Sac<Env, Mlp, Mlp2, ReplayBuffer> {
//!     let device = candle_core::Device::cuda_if_available(0).unwrap();
//!
//!     let actor_config = ActorConfig::default()
//!         .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
//!         .out_dim(out_dim)
//!         .pi_config(MlpConfig::new(in_dim, vec![64, 64], out_dim, true));
//!     let critic_config = CriticConfig::default()
//!         .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
//!         .q_config(MlpConfig::new(in_dim + out_dim, vec![64, 64], 1, true));
//!     let sac_config = SacConfig::<Mlp, Mlp2>::default()
//!         .batch_size(BATCH_SIZE)
//!         .actor_config(actor_config)
//!         .critic_config(critic_config)
//!         .device(device);
//!     Sac::build(sac_config)
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
