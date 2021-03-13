# Border

Border is a reinforcement learning library in Rust.

## Status

Border is currently under development.

## Prerequisites

In order to run examples, install python>=3.7 and [gym](https://gym.openai.com). Gym is the only built-in environment. The library itself works with any kind of environment.

## Examples

* Random agent: the following command runs a random agent for 5 episodes in [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/):

  ```bash
  $ cargo run --example random_cartpole
  ```

  It generates a csv file in `examples/model`, including the sequences of observation and reward values in the episodes.

## Features

* Environments which wrap [gym]() using [PyO3](https://crates.io/crates/pyo3) and [ndarray](https://crates.io/crates/ndarray)
* Interfaces to record quantities in training process or in evaluation path
  * Support tensorboard using [tensorboard-rs](https://crates.io/crates/tensorboard-rs)
* Vectorized environment using a tweaked [atari_wrapper.py](https://github.com/taku-y/border/blob/main/examples/atari_wrappers.py), adapted from the RL example in [tch](https://crates.io/crates/tch)
* Agents based on [tch](https://crates.io/crates/tch)
  * Currently including [DQN](https://arxiv.org/abs/1312.5602), [DDPG](https://arxiv.org/abs/1509.02971), [SAC](https://arxiv.org/abs/1801.01290)

## Roadmap

* More tests and documentations
* Improve performance (https://github.com/taku-y/border/issues/5)
* More environments
  * [pybullet-gym](https://github.com/benelot/pybullet-gym), [rogue-gym](https://github.com/kngwyu/rogue-gym), [ViZDoom](https://github.com/mwydmuch/ViZDoom), [gym-minecraft](https://github.com/tambetm/gym-minecraft)
* More RL algorithms
  * [DDQN](https://arxiv.org/abs/1509.06461), [Dueling network](https://arxiv.org/abs/1511.06581), [PPO](https://arxiv.org/abs/1707.06347), [QRDQN](https://arxiv.org/abs/1710.10044), [IQN](https://arxiv.org/abs/1806.06923), [TD3](https://arxiv.org/abs/1802.09477)
  * [Prioritized experience replay](https://arxiv.org/abs/1511.05952), [parameter noise](https://arxiv.org/abs/1706.01905)

## Licence

Border is primarily distributed under the terms of both the MIT license and the Apache License (Version 2.0).
