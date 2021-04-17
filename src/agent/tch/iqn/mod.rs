//! IQN agent.
pub mod base;
pub mod builder;
pub mod model;
pub mod explorer;
pub use base::IQN;
pub use builder::IQNBuilder;
pub use model::{IQNModelBuilder, IQNModel};
pub use explorer::{IQNExplorer, EpsilonGreedy};
