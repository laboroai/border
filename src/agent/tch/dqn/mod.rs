//! DQN agent.
#![allow(clippy::clippy::module_inception)]
pub mod dqn;
pub mod builder;
pub mod explorer;
pub use dqn::DQN;
pub use builder::DQNBuilder;
pub use explorer::{DQNExplorer, Softmax, EpsilonGreedy};