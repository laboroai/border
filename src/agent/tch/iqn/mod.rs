//! IQN agent.
pub mod base;
pub mod builder;
pub mod explorer;
pub mod model;
pub use base::IQN;
pub use builder::IQNBuilder;
pub use explorer::{EpsilonGreedy, IQNExplorer};
pub use model::{IQNModel, IQNModelBuilder};
