#![warn(missing_docs)]
//! Border is a library for reinforcement learning (RL).
pub mod core;
pub mod error;
mod shape;
pub use crate::core::{
    base::{Act, Agent, Env, Obs, Policy, Step, Info},
    trainer::{Trainer, TrainerBuilder},
    util::eval, util::eval_with_recorder,
    util,
    record,
};
pub use shape::Shape;
