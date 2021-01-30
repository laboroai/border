//! Primitives in reinforcement learning.
//!
//! This module does not related to concrete environments or algorithm implementations
//! with specific framework.
pub mod base;
pub use base::{Obs, Act, Info, Step, Env, Policy, Agent};
pub mod trainer;
pub use trainer::Trainer;
pub mod util;
