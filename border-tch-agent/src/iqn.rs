//! IQN agent.
mod base;
mod config;
mod explorer;
mod model;
pub use base::Iqn;
pub use config::IqnConfig;
pub use explorer::{EpsilonGreedy, IQNExplorer};
pub use model::{IqnModel, IqnModelConfig, IqnSample, average};
