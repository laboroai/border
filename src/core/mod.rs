//! Primitives in reinforcement learning.
//!
//! This module does not related to concrete environments or algorithm implementations
//! with specific framework.
pub mod base;
pub mod trainer;
pub mod util;
pub mod record;
pub use base::{Obs, Act, Info, Step, Env, Policy, Agent};
pub use trainer::{Trainer, TrainerBuilder};
