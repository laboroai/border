//! DQN agent.
mod base;
mod config;
mod explorer;
mod model;
pub use base::Dqn;
pub use config::DqnConfig;
pub use explorer::{DqnExplorer, EpsilonGreedy, Softmax};
pub use model::{DqnModel, DqnModelConfig};
