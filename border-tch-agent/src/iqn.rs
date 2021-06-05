//! IQN agent.
mod base;
mod builder;
mod explorer;
mod model;
pub use base::IQN;
pub use builder::IQNBuilder;
pub use explorer::{EpsilonGreedy, IQNExplorer};
pub use model::{IQNModel, IQNModelBuilder};
