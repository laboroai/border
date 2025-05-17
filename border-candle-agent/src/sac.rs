//! Soft actor-critic (SAC) agent.
mod base;
mod config;
mod ent_coef;
pub use base::Sac;
pub use config::SacConfig;
pub use ent_coef::{EntCoef, EntCoefMode};
