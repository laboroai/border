#![warn(missing_docs)]
//! Core components for reinforcement learning.
pub mod error;
pub mod record;
pub mod util;
pub mod replay_buffer;

mod base;
pub use base::{
    Act,
    Agent,
    BatchBase,
    Env,
    Info,
    Obs,
    Policy,
    ReplayBufferBase,
    Step,
    StepProcessorBase,
};

mod shape;
pub use shape::Shape;

mod trainer;
pub use trainer::{Trainer, TrainerConfig, SyncSampler};
