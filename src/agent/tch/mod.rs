//! Agents implemented based on tch-rs.
//!
//! This module includes agents, which implements some standard RL algorithms.
//! Currently, [DQN], [IQN], [SAC] are intensively developed,
//! alghough the module includes other algorithms [PPODiscrete], [pg], [DDPG].
//!
//! [ReplayBuffer] is used by agents in this module. It implements naive random sampling
//! from the internal buffer.
//!
//! # Design of agents
//!
//! ## Observation type for agents
//!
//! The agents are designed to be generic for observation type.
//! For example, [DQN] requires that its activation function takes a value of type
//! denoted by `Q::Input` in its trait bounds, and the observation type
//! of the environment ([border_core::Env::Obs]) is able to be converted to
//! `Q::Input` via [Into]. The observation of the environment, on which [DQN] works,
//! is bounded as `E::Obs: Into<Q::Input>`.
//!
//! ## Neural network models used by agents
//!
//! Each of [DQN], [IQN], and [SAC] has types of neural network models for its own.
//! For example, [DQN] uses [dqn::DQNModel] as the action value function.
//! The type of a model only defines its output type (`Q: SubModel<Output = Tensor>`) and
//! it can have a internal implementation specific to the problem.
//! `dqn_cartpole.rs` is a simple example, where the type `MLP` represents the action value function.
//! `MLP` implements [model::SubModel] trait for adapting it to [dqn::DQNModel].
//!
//! ## Configurations
//!
//! (not documented yet)
pub mod ddpg;
pub mod dqn;
pub mod iqn;
pub mod model;
pub mod opt;
pub mod pg;
pub mod ppo;
pub mod replay_buffer;
pub mod sac;
pub mod util;
pub use ddpg::DDPG;
pub use dqn::{DQNBuilder, DQN};
use ppo::ppo_discrete;
pub use ppo_discrete::PPODiscrete;
pub use replay_buffer::{ReplayBuffer, TchBatch, TchBuffer};
pub use sac::{SACBuilder, SAC};
pub use iqn::{IQN, IQNBuilder, model::{IQNModel, IQNModelBuilder}};
