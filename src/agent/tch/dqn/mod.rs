//! DQN agent.
#![allow(clippy::clippy::module_inception)]
pub mod builder;
pub mod dqn;
pub mod explorer;
pub use builder::DQNBuilder;
pub use dqn::DQN;
pub use explorer::{DQNExplorer, EpsilonGreedy, Softmax};
