//! Border is a reinforcement learning library.
//!
//! This crate is a collection of examples using the crates below.
//!
//! * [`border-core`](https://crates.io/crates/border-core) provides basic traits and functions
//!   generic to environments and reinforcmenet learning (RL) agents.
//! * [`border-py-gym-env`](https://crates.io/crates/border-py-gym-env) is a wrapper of the
//!   [Gym](https://gym.openai.com) environments written in Python, with the support of
//!   [pybullet-gym](https://github.com/benelot/pybullet-gym) and
//!   [atari](https://github.com/mgbellemare/Arcade-Learning-Environment).
//! * [`border-atari-env`](https://crates.io/crates/border-atari-env) is a wrapper of
//!   [atari-env](https://crates.io/crates/atari-env), which is a part of
//!   [gym-rs](https://crates.io/crates/gym-rs).
//! * [`border-tch-agent`](https://crates.io/crates/border-tch-agent) is a collection of RL agents
//!   based on [tch](https://crates.io/crates/tch). Deep Q network (DQN), implicit quantile network
//!   (IQN), and soft actor critic (SAC) are includes.
//! * [`border-async-trainer`](https://crates.io/crates/border-async-trainer) defines some traits and
//!   functions for asynchronous training of RL agents by multiple actors, each of which runs
//!   a sampling process of an agent and an environment in parallel.
//!
//! You can use a part of these crates for your purposes, though border-core is mandatory.
//!
//! # Evaluation
//!
//! For evaluating policies, [`border-core`](https://crates.io/crates/border-core) crate has the
//! following traits and structs with which RL problems are implemented.
//!
//! * [`Env`] provieds methods for environments.
//! * [`Env::Obs`], an associated type of [`Env`], represents the observation type.
//! * [`Env::Act`], an associated type of [`Env`], represents the action type.
//! * [`Policy`] is action distribution conditioned on observation.
//! * [`Recorder`] is used to record information during episodes.
//!
//! [`border-core`](https://crates.io/crates/border-core) crate has [`eval_with_recorder`]
//! function, which executes episodes with a policy on an environment and records
//! information.
//! See [`border_py_gym_env/examples/random_cartpols.rs`], which runs a random agent
//! on the cartpole environment.
//!
//! # Training
//!
//! You can train RL agents with the components above and those listed below.
//!
//! * [`Agent`] is a trainable [`Policy`].
//! * [`StepProcessorBase`] provieds methods for converting a [`Step<E: Env>`] object,
//!   representing a transition happens in `E`, and being pushed into a replay buffer
//!   implementing [`ReplayBufferBase`].
//! * [`BatchBase`] represents a batch generated from [`ReplayBufferBase`].
//!   Strictly, it is a trait that must be implemented on [`ReplayBufferBase::Batch`].
//!
//! [`Agent::opt()`] is supposed to implement a function that takes a mutable reference
//! to a replay buffer, implementing [`ReplayBufferBase`], and update model parameters
//! of the agent by using batches, implementing [`BatchBase`], taken from the replay buffer.
//!
//! See [`Trainer`](border_core::Trainer) for how these components interact.
//!
//! [`border-core`]: https://crates.io/crates/border-core
//! [`Env`]: border_core::Env
//! [`Env::Obs`]: border_core::Env::Obs
//! [`Env::Act`]: border_core::Env::Act
//! [`Policy`]: border_core::Policy
//! [`Recorder`]: border_core::record::Recorder
//! [`eval_with_recorder`]: border_core::util::eval_with_recorder
//! [`border_py_gym_env/examples/random_cartpols.rs`]: (https://github.com/taku-y/border/blob/982ef2d25a0ade93fb71cab3bb85e5062b6f769c/border-py-gym-env/examples/random_cartpole.rs)
//! [`Agent`]: border_core::Agent
//! [`StepProcessorBase`]: border_core::ReplayBufferBase
//! [`Step<E: Env>`]: border_core::Step
//! [`ReplayBufferBase`]: border_core::ReplayBufferBase
//! [`BatchBase`]: border_core::BatchBase
//! [`ReplayBufferBase::Batch`]: border_core::ReplayBufferBase::Batch
//! [`AAAgent::opt()`]: ../border_core/trait.Agent.html#tymethod.opt
//! [`Agent::opt()`]: border_core::Agent::opt

pub mod util;
