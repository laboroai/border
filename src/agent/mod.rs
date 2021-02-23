//! Agents of reinforcement learning.
//!
//! In this crate, an agent is defined as an trainable policy.
//! Currently, agents are implemented with [tch-rs](https://crates.io/crates/tch).
pub mod base;
pub mod tch;
pub use base::{OptInterval, OptIntervalCounter};
pub use self::tch::replay_buffer::{ReplayBuffer, TchBuffer};
pub use self::tch::dqn::DQN;
