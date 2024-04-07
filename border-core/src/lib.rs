#![warn(missing_docs)]
//! Core components for reinforcement learning.
//!
//! # Observation and action
//!
//! [`Obs`] and [`Act`] traits are abstractions of observation and action in environments.
//! These traits can handle two or more samples for implementing vectorized environments.
//!
//! # Environment
//!
//! [`Env`] trait is an abstraction of environments. It has four associated types:
//! `Config`, `Obs`, `Act` and `Info`. `Obs` and `Act` are concrete types of
//! observation and action of the environment.
//! These must implement [`Obs`] and [`Act`] traits, respectively.
//! The environment that implements [`Env`] generates [`Step<E: Env>`] object
//! at every environment interaction step with [`Env::step()`] method.
//!
//! `Info` stores some information at every step of interactions of an agent and
//! the environment. It could be empty (zero-sized struct). `Config` represents
//! configurations of the environment and is used to build.
//!
//! # Policy
//!
//! [`Policy<E: Env>`] represents a policy, from which actions are sampled for
//! environment `E`. [`Policy::sample()`] takes `E::Obs` and emits `E::Act`.
//! It could be probabilistic or deterministic.
//!
//! # Agent
//!
//! In this crate, [`Agent<E: Env, R: ReplayBufferBase>`] is defined as trainable
//! [`Policy<E: Env>`]. It is in either training or evaluation mode. In training mode,
//! the agent's policy might be probabilistic for exploration, while in evaluation mode,
//! the policy might be deterministic.
//!
//! [`Agent::opt()`] method does a single optimization step. The definition of an
//! optimization step depends on each agent. It might be multiple stochastic gradient
//! steps in an optimization step. Samples for training are taken from
//! [`R: ReplayBufferBase`][`ReplayBufferBase`].
//!
//! This trait also has methods for saving/loading the trained policy
//! in the given directory.
//!
//! # Replay buffer
//!
//! [`ReplayBufferBase`] trait is an abstraction of replay buffers. For handling samples,
//! there are two associated types: `PushedItem` and `Batch`. `PushedItem` is a type
//! representing samples pushed to the buffer. These samples might be generated from
//! [`Step<E: Env>`]. [`StepProcessorBase<E: Env>`] trait provides the interface
//! for converting [`Step<E: Env>`] into `PushedItem`.
//!
//! `Batch` is a type of samples taken from the buffer for training [`Agent`]s.
//! The user implements [`Agent::opt()`] method such that it handles `Batch` objects
//! for doing an optimization step.
//!
//! ## A reference implementation
//!
//! [`SimpleReplayBuffer<O, A>`] implementats [`ReplayBufferBase`].
//! This type has two parameters `O` and `A`, which are representation of
//! observation and action in the replay buffer. `O` and `A` must implement
//! [`SubBatch`], which has the functionality of storing samples, like `Vec<T>`,
//! for observation and action. The associated types `PushedItem` and `Batch`
//! are the same type, [`StdBatch`], representing sets of `(o_t, r_t, a_t, o_t+1)`.
//!
//! [`SimpleStepProcessor<E, O, A>`] might be used with [`SimpleReplayBuffer<O, A>`].
//! It converts `E::Obs` and `E::Act` into [`SubBatch`]s of respective types
//! and generates [`StdBatch`]. The conversion process relies on trait bounds,
//! `O: From<E::Obs>` and `A: From<E::Act>`.
//!
//! # Trainer
//!
//! [`Trainer`] manages training loop and related objects. The [`Trainer`] object is
//! built with configurations of [`Env`], [`ReplayBufferBase`], [`StepProcessorBase`]
//! and some training parameters. Then, [`Trainer::train`] method starts training loop with
//! given [`Agent`] and [`Recorder`](crate::record::Recorder).
//!
//! [`SimpleReplayBuffer`]: replay_buffer::SimpleReplayBuffer
//! [`SimpleReplayBuffer<O, A>`]: replay_buffer::SimpleReplayBuffer
//! [`SubBatch`]: replay_buffer::SubBatch
//! [`StdBatch`]: replay_buffer::StdBatch
//! [`SimpleStepProcessor`]: replay_buffer::SimpleStepProcessor
//! [`SimpleStepProcessor<E, O, A>`]: replay_buffer::SimpleStepProcessor
pub mod error;
mod evaluator;
pub mod record;
pub mod replay_buffer;
pub mod util;

mod base;
pub use base::{
    Act, Agent, Env, ExperienceBufferBase, Info, Obs, Policy, ReplayBufferBase, StdBatchBase, Step,
    StepProcessorBase,
};

mod trainer;
pub use evaluator::{DefaultEvaluator, Evaluator};
pub use trainer::{SyncSampler, Trainer, TrainerConfig};
