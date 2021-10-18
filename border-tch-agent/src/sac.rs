//! SAC agent.
mod actor;
mod base;
mod config;
mod critic;
mod ent_coef;
pub use actor::{Actor, ActorConfig};
pub use base::SAC;
pub use config::SACConfig;
pub use critic::{Critic, CriticConfig};
pub use ent_coef::{EntCoef, EntCoefMode};
