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
//! # Batch
//!
//! [`TransitionBatch`] is a trait of a batch of transitions `(o_t, r_t, a_t, o_t+1)`.
//! This is used to train [`Agent`]s with an RL algorithm.
//!
//! # Replay buffer
//!
//! [`ReplayBufferBase`] trait is an abstraction of replay buffers. For handling samples,
//! there are two associated types: `Item` and `Batch`. `Item` is a type
//! representing samples pushed to the buffer. These samples might be generated from
//! [`Step<E: Env>`]. [`StepProcessor<E: Env>`] trait provides the interface
//! for converting [`Step<E: Env>`] into `Item`.
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
//! [`BatchBase`], which has the functionality of storing samples, like `Vec<T>`,
//! for observation and action. The associated types `Item` and `Batch`
//! are the same type, [`GenericTransitionBatch`], representing sets of `(o_t, r_t, a_t, o_t+1)`.
//!
//! [`SimpleStepProcessor<E, O, A>`] might be used with [`SimpleReplayBuffer<O, A>`].
//! It converts `E::Obs` and `E::Act` into [`BatchBase`]s of respective types
//! and generates [`GenericTransitionBatch`]. The conversion process relies on trait bounds,
//! `O: From<E::Obs>` and `A: From<E::Act>`.
//!
//! # Trainer
//!
//! [`Trainer`] manages training loop and related objects. The [`Trainer`] object is
//! built with configurations of [`Env`], [`ReplayBufferBase`], [`StepProcessor`]
//! and some training parameters. Then, [`Trainer::train`] method starts training loop with
//! given [`Agent`] and [`Recorder`](crate::record::Recorder).
//!
//! [`SimpleReplayBuffer`]: replay_buffer::SimpleReplayBuffer
//! [`SimpleReplayBuffer<O, A>`]: generic_replay_buffer::SimpleReplayBuffer
//! [`BatchBase`]: generic_replay_buffer::BatchBase
//! [`GenericTransitionBatch`]: generic_replay_buffer::GenericTransitionBatch
//! [`SimpleStepProcessor`]: replay_buffer::SimpleStepProcessor
//! [`SimpleStepProcessor<E, O, A>`]: generic_replay_buffer::SimpleStepProcessor
pub mod error;
mod evaluator;
pub mod generic_replay_buffer;
pub mod record;

mod base;
pub use base::{
    Act, Agent, Configurable, Env, ExperienceBufferBase, Info, Obs, Policy, ReplayBufferBase, Step,
    StepProcessor, TransitionBatch,
};

mod trainer;
pub use evaluator::{DefaultEvaluator, Evaluator};
pub use trainer::{Sampler, Trainer, TrainerConfig};
