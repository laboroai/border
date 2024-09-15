# Border

A reinforcement learning library in Rust.

[![CI](https://github.com/taku-y/border/actions/workflows/ci.yml/badge.svg)](https://github.com/taku-y/border/actions/workflows/ci.yml)
[![Latest version](https://img.shields.io/crates/v/border.svg)](https://crates.io/crates/border)
[![Documentation](https://docs.rs/border/badge.svg)](https://docs.rs/border)
![License](https://img.shields.io/crates/l/border.svg)

Border consists of the following crates:

* Core and utility
  * [border-core](https://crates.io/crates/border-core) ([doc](https://docs.rs/border-core/latest/border_core/)) provides basic traits and functions generic to environments and reinforcmenet learning (RL) agents.
  * [border-tensorboard](https://crates.io/crates/border-tensorboard) ([doc](https://docs.rs/border-core/latest/border_tensorboard/)) has `TensorboardRecorder` struct to write records which can be shown in Tensorboard. It is based on [tensorboard-rs](https://crates.io/crates/tensorboard-rs).
  * [border-mlflow-tracking](https://crates.io/crates/border-mlflow-tracking) ([doc](https://docs.rs/border-core/latest/border_mlflow_tracking/)) support MLflow tracking to log metrices during training via REST API.
  * [border-async-trainer](https://crates.io/crates/border-async-trainer) ([doc](https://docs.rs/border-core/latest/border_async_trainer/)) defines some traits and functions for asynchronous training of RL agents by multiple actors, which runs sampling processes in parallel. In each sampling process, an agent interacts with an environment to collect samples to be sent to a shared replay buffer.
  * [border](https://crates.io/crates/border) is just a collection of examples.
* Environment
  * [border-py-gym-env](https://crates.io/crates/border-py-gym-env) ([doc](https://docs.rs/border-core/latest/border_py_gym_env/)) is a wrapper of the [Gymnasium](https://gymnasium.farama.org) environments written in Python.
  * [border-atari-env](https://crates.io/crates/border-atari-env) ([doc](https://docs.rs/border-core/latest/border_atari_env/)) is a wrapper of [atari-env](https://crates.io/crates/atari-env), which is a part of [gym-rs](https://crates.io/crates/gym-rs).
  * [border-minari](https://crates.io/crates/border-minari) ([doc](https://docs.rs/border-core/latest/border_minari/))
    is a wrapper of [Minari](https://minari.farama.org).
* Agent
  * [border-tch-agent](https://crates.io/crates/border-tch-agent) ([doc](https://docs.rs/border-core/latest/border_tch_agent/)) includes RL agents based on [tch](https://crates.io/crates/tch), including Deep Q network (DQN), implicit quantile network (IQN), and soft actor critic (SAC).
  * [border-candle-agent](https://crates.io/crates/border-candle-agent) ([doc](https://docs.rs/border-core/latest/border_candle_agent/)) includes RL agents based on [candle](https://crates.io/crates/candle-core)
  * [border-policy-no-backend](https://crates.io/crates/border-policy-no-backend) ([doc](https://docs.rs/border-core/latest/border_policy_no_backend/)) includes a policy that is independent of any deep learning backend, such as Torch.

## Status

Border is experimental and currently under development. API is unstable.

## Examples

There are some example sctipts in `border/examples` directory. These are tested in Docker containers, speficically the one in `aarch64` directory on M2 Macbook air. Some scripts take few days for the training process, tested on Ubuntu22.04 virtual machine in  [GPUSOROBAN](https://soroban.highreso.jp), a computing cloud.

## Docker

In `docker` directory, there are scripts for running a Docker container, in which you can try the examples described above. Currently, only `aarch64` is mainly used for the development.

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
