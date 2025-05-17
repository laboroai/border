# Border

A reinforcement learning library in Rust.

[![CI](https://github.com/taku-y/border/actions/workflows/ci.yml/badge.svg)](https://github.com/taku-y/border/actions/workflows/ci.yml)
[![Latest version](https://img.shields.io/crates/v/border.svg)](https://crates.io/crates/border)
[![Documentation](https://docs.rs/border/badge.svg)](https://docs.rs/border)
![License](https://img.shields.io/crates/l/border.svg)

Border consists of the following crates:

* Core and utility
  * [border-core](https://crates.io/crates/border-core) ([doc](https://docs.rs/border-core/latest/border_core/)) provides basic traits and functions for environments and reinforcement learning (RL) agents.
  * [border-tensorboard](https://crates.io/crates/border-tensorboard) ([doc](https://docs.rs/border-core/latest/border_tensorboard/)) implements the `TensorboardRecorder` struct for writing records that can be visualized in Tensorboard, based on [tensorboard-rs](https://crates.io/crates/tensorboard-rs).
  * [border-mlflow-tracking](https://crates.io/crates/border-mlflow-tracking) ([doc](https://docs.rs/border-core/latest/border_mlflow_tracking/)) provides MLflow tracking support for logging metrics during training via REST API.
  * [border-async-trainer](https://crates.io/crates/border-async-trainer) ([doc](https://docs.rs/border-core/latest/border_async_trainer/)) defines traits and functions for asynchronous training of RL agents using multiple actors. Each actor runs a sampling process in parallel, where an agent interacts with an environment to collect samples for a shared replay buffer.
  * [border](https://crates.io/crates/border) serves as a collection of examples.
* Environment
  * [border-py-gym-env](https://crates.io/crates/border-py-gym-env) ([doc](https://docs.rs/border-core/latest/border_py_gym_env/)) provides a wrapper for [Gymnasium](https://gymnasium.farama.org) environments written in Python.
  * [border-atari-env](https://crates.io/crates/border-atari-env) ([doc](https://docs.rs/border-core/latest/border_atari_env/)) implements a wrapper for [atari-env](https://crates.io/crates/atari-env), which is part of [gym-rs](https://crates.io/crates/gym-rs).
  * [border-minari](https://crates.io/crates/border-minari) ([doc](https://docs.rs/border-core/latest/border_minari/)) provides a wrapper for [Minari](https://minari.farama.org).
* Agent
  * [border-tch-agent](https://crates.io/crates/border-tch-agent) ([doc](https://docs.rs/border-core/latest/border_tch_agent/)) implements RL agents based on [tch](https://crates.io/crates/tch), including Deep Q Network (DQN), Implicit Quantile Network (IQN), and Soft Actor-Critic (SAC).
  * [border-candle-agent](https://crates.io/crates/border-candle-agent) ([doc](https://docs.rs/border-core/latest/border_candle_agent/)) implements RL agents based on [candle](https://crates.io/crates/candle-core).
  * [border-policy-no-backend](https://crates.io/crates/border-policy-no-backend) ([doc](https://docs.rs/border-core/latest/border_policy_no_backend/)) implements policies that are independent of any deep learning backend, such as Torch.

## Status

Border is experimental and currently under development. The API is unstable.

## Examples

Example scripts are available in the `examples` directory. These have been tested in Docker containers. Some scripts require several days for the training process, as tested on an Ubuntu 22.04 virtual machine.

## Docker

Docker configuration files for development and testing are available in the [dev-border](https://github.com/taku-y/dev-border) repository. These files are used to set up the development environment, supporting both aarch64 (e.g., M2 MacBook Air) and amd64 architectures.

## License

Crates                    | License
--------------------------|------------------
`border-core`             | MIT OR Apache-2.0
`border-tensorboard`      | MIT OR Apache-2.0
`border-mlflow-tracking`  | MIT OR Apache-2.0
`border-async-trainer`    | MIT OR Apache-2.0
`border-py-gym-env`       | MIT OR Apache-2.0
`border-atari-env`        | GPL-2.0-or-later
`border-minari`           | MIT OR Apache-2.0
`border-tch-agent`        | MIT OR Apache-2.0
`border-candle-agent`     | MIT OR Apache-2.0
`border-policy-no-backend`| MIT OR Apache-2.0
`border`                  | GPL-2.0-or-later
