#![warn(missing_docs)]
//! Core components for reinforcement learning.
pub mod error;
pub mod record;
pub mod replay_buffer;
pub mod util;

mod base;
pub use base::{
    Act, Agent, StdBatchBase, Env, ExperienceBufferBase, PushedItemBase, Info, Obs, Policy, ReplayBufferBase,
    ReplayBufferBaseConfig, Step, StepProcessorBase,
};

mod shape;
pub use shape::Shape;

mod trainer;
pub use trainer::{SyncSampler, Trainer, TrainerConfig};
