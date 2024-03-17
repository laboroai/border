# Border

A reinforcement learning library in Rust.

[![CI](https://github.com/taku-y/border/actions/workflows/ci.yml/badge.svg)](https://github.com/taku-y/border/actions/workflows/ci.yml)
[![Latest version](https://img.shields.io/crates/v/border.svg)](https://crates.io/crates/border)
[![Documentation](https://docs.rs/border/badge.svg)](https://docs.rs/border)
![License](https://img.shields.io/crates/l/border.svg)

Border consists of the following crates:

* [border-core](https://crates.io/crates/border-core) provides basic traits and functions generic to environments and reinforcmenet learning (RL) agents.
* [border-tensorboard](https://crates.io/crates/border-tensorboard) has `TensorboardRecorder` struct to write records which can be shown in Tensorboard. It is based on [tensorboard-rs](https://crates.io/crates/tensorboard-rs).
* [border-py-gym-env](https://crates.io/crates/border-py-gym-env) is a wrapper of the [Gymnasium](https://gymnasium.farama.org) environments written in Python.
* [border-atari-env](https://crates.io/crates/border-atari-env) is a wrapper of [atari-env](https://crates.io/crates/atari-env), which is a part of [gym-rs](https://crates.io/crates/gym-rs).
* [border-tch-agent](https://crates.io/crates/border-tch-agent) is a collection of RL agents based on [tch](https://crates.io/crates/tch), including Deep Q network (DQN), implicit quantile network (IQN), and soft actor critic (SAC).
* [border-async-trainer](https://crates.io/crates/border-async-trainer) defines some traits and functions for asynchronous training of RL agents by multiple actors, which runs sampling processes in parallel. In each sampling process, an agent interacts with an environment to collect samples to be sent to a shared replay buffer.
* [border-mlflow-tracking](https://crates.io/crates/border-mlflow-tracking) support MLflow tracking to log metrices during training via REST API.

You can use a part of these crates for your purposes, though [border-core](https://crates.io/crates/border-core) is mandatory. [This crate](https://crates.io/crates/border) is just a collection of examples. See [Documentation](https://docs.rs/border) for more details.

## News

The owner of this repository will be changed from [taku-y](https://github.com/taku-y) to [laboroai](https://github.com/laboroai).

## Status

Border is experimental and currently under development. API is unstable.

## Examples

There are some example sctipts in `border/examples` directory. These are tested in Docker containers, speficically the one in `aarch64` directory on M2 Macbook air. Some scripts take few days for the training process, tested on Ubuntu22.04 virtual machine in  [GPUSOROBAN](https://soroban.highreso.jp), a computing cloud.

## Docker

In `docker` directory, there are scripts for running a Docker container, in which you can try the examples described above. Currently, only `aarch64` is mainly used for the development.

## Tests

The following command has been tested in the Docker container running on M2 Macbook air.

```bash
cargo test --features=tch
```

## License

Crates                | License
----------------------|------------------
`border-core`         | MIT OR Apache-2.0
`border-py-gym-env`   | MIT OR Apache-2.0
`border-atari-env`    | GPL-2.0-or-later
`border-tch-agent`    | MIT OR Apache-2.0
`border-async-trainer`| MIT OR Apache-2.0
`border`              | GPL-2.0-or-later
