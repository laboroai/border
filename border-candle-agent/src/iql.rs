//! Implicit Q-learning agent.
mod base;
mod config;
mod value;
pub use base::Iql;
pub use config::IqlConfig;
pub use value::{Value, ValueConfig};
