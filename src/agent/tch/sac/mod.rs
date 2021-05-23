//! SAC agent.
pub mod base;
pub mod builder;
pub mod ent_coef;
pub mod actor;
pub mod critic;
pub use base::SAC;
pub use builder::SACBuilder;
pub use ent_coef::{EntCoef, EntCoefMode};
