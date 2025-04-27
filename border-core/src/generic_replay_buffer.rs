//! Generic implementation of replay buffers for reinforcement learning.
//!
//! This module provides a flexible and efficient implementation of replay buffers
//! that can handle arbitrary observation and action types. It supports both
//! standard experience replay and prioritized experience replay (PER).
//!
//! # Key Components
//!
//! - [`SimpleReplayBuffer`]: A generic replay buffer implementation
//! - [`GenericTransitionBatch`]: A generic batch structure for transitions
//! - [`SimpleStepProcessor`]: A processor for converting environment steps to transitions
//! - [`PerConfig`]: Configuration for prioritized experience replay
//!
//! # Features
//!
//! - Generic type support for observations and actions
//! - Efficient batch processing
//! - Prioritized experience replay with importance sampling
//! - Configurable weight normalization
//! - Step processing for non-vectorized environments

mod base;
mod batch;
mod config;
mod step_proc;
pub use base::{IwScheduler, SimpleReplayBuffer, WeightNormalizer};
pub use batch::{BatchBase, GenericTransitionBatch};
pub use config::{PerConfig, SimpleReplayBufferConfig};
pub use step_proc::{SimpleStepProcessor, SimpleStepProcessorConfig};
