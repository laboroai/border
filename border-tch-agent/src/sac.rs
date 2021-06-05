//! SAC agent.
mod actor;
mod base;
mod builder;
mod critic;
mod ent_coef;
pub use actor::{Actor, ActorBuilder};
pub use base::SAC;
pub use builder::SACBuilder;
pub use critic::{Critic, CriticBuilder};
pub use ent_coef::{EntCoef, EntCoefMode};
