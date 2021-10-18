//! DQN agent.
mod base;
mod config;
mod explorer;
mod model;
pub use base::DQN;
pub use config::DQNConfig;
pub use explorer::{DQNExplorer, EpsilonGreedy, Softmax};
pub use model::{DQNModel, DQNModelConfig};
