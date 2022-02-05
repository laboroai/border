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
//! # Evaluation
//!
//! For evaluating policies, [`border-core`](https://crates.io/crates/border-core) crate has the
//! following traits on which RL problems are implemented.
//!
//! * [`Env`](border_core::Env) - environments.
//! * [`Env::Obs`](border_core::Env::Obs) - observation of the environment.
//! * [`Env::Act`](border_core::Env::Act) - action of the environment.
//! * [`Policy`](border_core::Policy) - action distribution conditioned on observation.
//! * [`Recorder`](border_core::record::Recorder) - recording information during running episodes.
//!
//! [`border-core`](https://crates.io/crates/border-core) crate has
//! [`eval_with_recorder`](border_core::util::eval_with_recorder) function, which executes episodes
//! with a policy on an environment and records information.
//! See [`border_py_gym_env/examples/random_cartpols.rs`](https://github.com/taku-y/border/blob/982ef2d25a0ade93fb71cab3bb85e5062b6f769c/border-py-gym-env/examples/random_cartpole.rs)
//! which runs a random agent on the cartpole environment.
//!
//! # Training
//!
//! For training RL agents, [`border-core`](https://crates.io/crates/border-core) crate also has
//! components.
//!
//! * [`Agent`](border_core::Agent) - trainable RL agents, inheritating [`Policy`](border_core::Policy).
//! * [`StepProcessorBase`](border_core::ReplayBufferBase) - making
//!   [`Batch`](border_core::Batch), which will be fed into a replay buffer.
//! * [`ReplayBufferBase`](border_core::ReplayBufferBase) - replay buffers.
//!
//! See [`Trainer`](border_core::Trainer) for how these components interact.

pub mod util;
