#![warn(missing_docs)]
//! A library for reinforcement learning.
pub mod error;
pub mod record;
pub mod util;
pub mod replay_buffer;

mod base;
pub use base::{
    Act,
    Agent,
    Batch,
    Env,
    Info,
    Obs,
    Policy,
    ReplayBufferBase,
    Step,
    StepProcessorBase,
    // trainer::{Trainer, TrainerBuilder},
    // util::eval, util::eval_with_recorder,
    // util,
    // record,
};

mod shape;
pub use shape::Shape;

mod trainer;
pub use trainer::{Trainer, TrainerConfig, SyncSampler};
