//! SAC agent.
pub mod actor;
pub mod base;
pub mod builder;
pub mod critic;
pub mod ent_coef;
pub use actor::{Actor, ActorBuilder};
pub use base::SAC;
pub use builder::SACBuilder;
pub use critic::{Critic, CriticBuilder};
pub use ent_coef::{EntCoef, EntCoefMode};
