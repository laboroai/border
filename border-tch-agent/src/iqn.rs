//! IQN agent.
mod base;
mod config;
mod explorer;
mod model;
pub use base::Iqn;
pub use config::IqnConfig;
pub use explorer::{EpsilonGreedy, IqnExplorer, Softmax};
pub use model::{average, IqnModel, IqnModelConfig, IqnSample};
