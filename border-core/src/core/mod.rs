//! Primitives in reinforcement learning.
//!
//! This module does not related to concrete environments or algorithm implementations
//! with specific framework.
pub mod base;
pub mod record;
pub mod trainer;
pub mod util;
pub use base::{Act, Agent, Env, Info, Obs, Policy, Step};
pub use trainer::{Trainer, TrainerBuilder};
