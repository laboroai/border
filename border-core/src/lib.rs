#![warn(missing_docs)]
//! Core components for reinforcement learning.
//!
//! # Observation and action
//!
//! [`Obs`] and [`Act`] traits are abstractions of observation and action in environment$.
//! These traits can handle two or more samples for implementing vectorized environments.
//!
//! # Environment
//!
//! [`Env`] trait is an abstraction of environments. It has four associated types:
//! `Config`, `Obs`, `Act` and `Info`. `Obs` and `Act` are concrete types of
//! observation and action of the environment. 
//! These must implement [`Obs`] and [`Act`] traits, respectively.
//! The environment that implements [`Env`] generates [`Step`]`<Self>` object
//! at every environment interaction step with a [`Policy`].
//! 
//! `Info` can store some information at every step of interactions of an agent and
//! the environment. It could be empty (zero-sized struct). `Config` represents
//! configurations of the environment and is used to build.
//!
//! # Policy
//!
//! [`Policy`] represents a policy, from which actions are sampled.
//! It could be probabilistic or deterministic.
//! `Policy` has a type parameter `E: `[`Env`], with which the policy interacts.
//!
//! # Agent
//!
//! In this crate, [`Agent`] is defined as trainable [`Policy`]. It is either training or
//! evaluation mode. In training mode, the agent's policy might be probabilistic for
//! exploration, while in evaluation mode, the policy might be deterministic.
//!
//! `opt()` method does a single optimization step. The definition of an optimization step
//! depends on each agent. It does multiple stochastic gradient steps in an optimization step.
//! Samples for training are taken from `R: `[`ReplayBufferBase`].
//!
//!  This trait also has methods for saving/loading the trained policy, which is usually
//! a set of model parameters of the policy.
//! 
//! # Replay buffer
//! 
//! [`ReplayBufferBase`] trait is an abstraction of replay buffers. For handling samples,
//! there are two associated types: `PushedItem` and `Batch`. `PushedItem` is a type
//! representing samples pushed to the buffer. These samples might be generated from
//! [`Step`]`<E: `[`Env`]`>`. [`StepProcessorBase`] trait provides the interface for converting
//! [`Step`]`<E: `[`Env`]`>` into `PushedItem`.
//! 
//! `Batch` is a type of samples taken from the buffer for training [`Agent`]s.
//! The user implements `opt()` method in `Agent` such that it handles `Batch` objects
//! for doing an optimization step.
//! 
//! # Trainer
//! 
//! [`Trainer`] manages training loop and related objects. The object is built with
//! configurations of [`Env`], [`ReplayBufferBase`], [`StepProcessorBase`] and some
//! training parameters. Then, `train()` method starts training loop with given
//! [`Agent`] and [`Recorder`](crate::record::Recorder).
pub mod error;
pub mod record;
pub mod replay_buffer;
pub mod util;

mod base;
pub use base::{
    Act, Agent, Env, ExperienceBufferBase, Info, Obs, Policy, ReplayBufferBase, StdBatchBase, Step,
    StepProcessorBase,
};

mod shape;
pub use shape::Shape;

mod trainer;
pub use trainer::{SyncSampler, Trainer, TrainerConfig};
