# Border

A reinforcement learning library in Rust.

[![CI](https://github.com/taku-y/border/actions/workflows/ci.yml/badge.svg)](https://github.com/taku-y/border/actions/workflows/ci.yml)
[![Latest version](https://img.shields.io/crates/v/border.svg)](https://crates.io/crates/border)
[![Documentation](https://docs.rs/border/badge.svg)](https://docs.rs/border)
![License](https://img.shields.io/crates/l/border.svg)

Border consists of the following crates:

* [border-core](https://crates.io/crates/border-core) provides basic traits and functions generic to environments and reinforcmenet learning (RL) agents.
* [border-py-gym-env](https://crates.io/crates/border-py-gym-env) is a wrapper of the [Gym](https://gym.openai.com) environments written in Python, with the support of [pybullet-gym](https://github.com/benelot/pybullet-gym) and [atari](https://github.com/mgbellemare/Arcade-Learning-Environment).
* [border-atari-env](https://crates.io/crates/border-atari-env) is a wrapper of [atari-env](https://crates.io/crates/atari-env), which is a part of [gym-rs](https://crates.io/crates/gym-rs).
* [border-tch-agent](https://crates.io/crates/border-tch-agent) is a collection of RL agents based on [tch](https://crates.io/crates/tch). Deep Q network (DQN), implicit quantile network (IQN), and soft actor critic (SAC) are includes.
* [border-async-trainer](https://crates.io/crates/border-async-trainer) defines some traits and functions for asynchronous training of RL agents by multiple actors, each of which runs a sampling process of an agent and an environment in parallel.

You can use a part of these crates for your purposes, though [border-core](https://crates.io/crates/border-core) is mandatory. [This crate](https://crates.io/crates/border) is just a collection of examples. See [Documentation](https://docs.rs/border) for more details.

## Status

Border is experimental and currently under development. API is unstable.

## Examples

In examples directory, you can see how to run some examples. Python>=3.7 and [gym](https://gym.openai.com) must be installed for running examples using [border-py-gym-env](https://crates.io/crates/border-py-gym-env). Some examples requires [PyBullet Gym](https://github.com/benelot/pybullet-gym). As the agents used in the examples are based on [tch-rs](https://github.com/LaurentMazare/tch-rs), libtorch is required to be installed.

## License

Crates                | License
----------------------|------------------
`border-core`         | MIT OR Apache-2.0
`border-py-gym-env`   | MIT OR Apache-2.0
`border-atari-env`    | GPL-2.0-or-later
`border-tch-agent`    | MIT OR Apache-2.0
`border-async-trainer`| MIT OR Apache-2.0
`border`              | GPL-2.0-or-later
