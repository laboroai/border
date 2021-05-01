//! Agents implemented based on tch-rs.
pub mod ddpg;
pub mod dqn;
pub mod iqn;
pub mod model;
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
// pub use iqn::{IQN, IQNBuilder, model::{IQNModel, IQNModelBuilder}};
