//! SAC agent.
pub mod actor;
pub mod base;
pub mod builder;
pub mod critic;
pub mod ent_coef;
pub use base::SAC;
pub use builder::SACBuilder;
pub use ent_coef::{EntCoef, EntCoefMode};
