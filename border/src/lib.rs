//! A reinforcement learning library in Rust.
//!
//! Border consists of the following crates:
//!
//! * Core and utility
//!   * [border-core](https://crates.io/crates/border-core) provides basic traits 
//!     and functions generic to environments and reinforcmenet learning (RL) agents.
//!   * [border-tensorboard](https://crates.io/crates/border-tensorboard) has 
//!     `TensorboardRecorder` struct to write records which can be shown in Tensorboard.
//!     It is based on [tensorboard-rs](https://crates.io/crates/tensorboard-rs).
//!   * [border-mlflow-tracking](https://crates.io/crates/border-mlflow-tracking)
//!     support MLflow tracking to log metrices during training via REST API.
//!   * [border-async-trainer](https://crates.io/crates/border-async-trainer) defines
//!     some traits and functions for asynchronous training of RL agents by multiple
//!     actors, which runs sampling processes in parallel. In each sampling process,
//!     an agent interacts with an environment to collect samples to be sent to a shared
//!     replay buffer.
//!   * [border](https://crates.io/crates/border) is just a collection of examples.
//! * Environment
//!   * [border-py-gym-env](https://crates.io/crates/border-py-gym-env) is a wrapper of the
//!     [Gymnasium](https://gymnasium.farama.org) environments written in Python.
//!   * [border-atari-env](https://crates.io/crates/border-atari-env) is a wrapper of
//!     [atari-env](https://crates.io/crates/atari-env), which is a part of
//!     [gym-rs](https://crates.io/crates/gym-rs).
//! * Agent
//!   * [border-tch-agent](https://crates.io/crates/border-tch-agent) includes RL agents
//!     based on [tch](https://crates.io/crates/tch), including Deep Q network (DQN),
//!     implicit quantile network (IQN), and soft actor critic (SAC).
//!   * [border-candle-agent](https://crates.io/crates/border-candle-agent) includes RL
//!     agents based on [candle](https://crates.io/crates/candle-core)
//!   * [border-policy-no-backend](https://crates.io/crates/border-policy-no-backend)
//!     includes a policy that is independent of any deep learning backend, such as Torch.

pub mod util;
