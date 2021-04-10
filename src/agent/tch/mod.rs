//! Agents implemented based on tch-rs.
pub mod util;
pub mod replay_buffer;
pub mod dqn;
pub mod ppo;
pub mod sac;
pub mod ddpg;
pub mod pg;
pub mod model;
pub mod iqn;
use ppo::ppo_discrete;
pub use replay_buffer::{ReplayBuffer, TchBuffer, TchBatch};
pub use dqn::{DQN, DQNBuilder};
pub use ppo_discrete::PPODiscrete;
pub use sac::{SAC, SACBuilder};
pub use ddpg::DDPG;
pub use iqn::{IQNAgent, IQNBuilder};
