# Border

Border is a reinforcement learning library in Rust.

## Status

Border is currently under development.

## Prerequisites

In order to run examples, install python>=3.7 and [gym](https://gym.openai.com), for which the library provides a wrapper using [PyO3](https://crates.io/crates/pyo3).

## Examples

### Random policy on cartople environment

The following command runs a random controller (policy) for 5 episodes in [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/):

  ```bash
  $ cargo run --example random_cartpole
  ```

  It renders during the episodes and generates a csv file in `examples/model`, including the sequences of observation and reward values in the episodes.

  ```bash
  $ head -n3 examples/model/random_cartpole_eval.csv
  0,0,1.0,-0.012616985477507114,0.19292789697647095,0.04204097390174866,-0.2809212803840637
  0,1,1.0,-0.008758427575230598,-0.0027677505277097225,0.036422546952962875,0.024719225242733955
  0,2,1.0,-0.008813782595098019,-0.1983925849199295,0.036916933953762054,0.3286677300930023
  ```

### Deep Q-network (DQN) on cartpole environment

The following command trains a DQN agent:

  ```bash
  $ cargo run --example dqn_cartpole
  ```

  After training, the trained agent runs for 5 episodes. The parameters of the trained Q-network (and the target network) are saved in `examples/model/dqn_cartpole`.

### Soft actor-critic (SAC) on pendulum environment

The following command trains a SAC agent on [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/), which takes continuous action:

  ```bash
  $ cargo run --example sac_pendulum
  ```

  The code defines an action filter that doubles the torque in the environment.

### Atari games

The following command trains a DQN agent on [PongNoFrameskip-v4](https://gym.openai.com/envs/Pong-v0/):

  ```bash
  $ PYTHONPATH=$REPO/examples cargo run --release --example dqn_atari -- PongNoFrameskip-v4
  ```

During training, the program will save the model parameters when the evaluation reward achieves its maximum value. The agent can be trained for other atari games (e.g., `SeaquestNoFrameskip-v4`) by replacing the name of the environment in the above command.

For Pong, you can download a pretrained agent from my google drive and see how it plays with the following command:

  ```bash
  $ PYTHONPATH=$REPO/examples cargo run --release --example dqn_atari -- PongNoFrameskip-v4 --play-gdrive
  ```

The pretrained agent will be saved locally in `$HOME/.border/model`.

### Vectorized environment for atari games

The following command trains a DQN agent in an vectorized environment of Pong:

  ```bash
  $ PYTHONPATH=$REPO/examples cargo run --release --example dqn_pong_vecenv
  ```

  The code demonstrates how to use vectorized environments, in which 4 environments are running synchronously. It took about 11 hours for 2M steps (8M transition samples) on a `g3s.xlarge` instance of EC2. Hyperparameter values, tuned specific to Pong instead of all Atari games, are adapted from the book [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994). The learning curve is as shown below.

  <img src="learning_curve.png" width="256">

  After the training, you can see how the agent plays:

  ```bash
  $ PYTHONPATH=$REPO/examples cargo run --example dqn_pong_eval
  ```

## Features

* Environments which wrap [gym](https://gym.openai.com) using [PyO3](https://crates.io/crates/pyo3) and [ndarray](https://crates.io/crates/ndarray)
* Interfaces to record quantities in training process or in evaluation path
  * Support tensorboard using [tensorboard-rs](https://crates.io/crates/tensorboard-rs)
* Vectorized environment using a tweaked [atari_wrapper.py](https://github.com/taku-y/border/blob/main/examples/atari_wrappers.py), adapted from the RL example in [tch](https://crates.io/crates/tch)
* Agents based on [tch](https://crates.io/crates/tch)
  * Currently including [DQN](https://arxiv.org/abs/1312.5602), [Deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971), [SAC](https://arxiv.org/abs/1801.01290), [Implicit quantile network (IQN)](https://arxiv.org/abs/1806.06923).

## Roadmap

* More tests and documentations
* More environments
  * [pybullet-gym](https://github.com/benelot/pybullet-gym), [rogue-gym](https://github.com/kngwyu/rogue-gym), [ViZDoom](https://github.com/mwydmuch/ViZDoom), [gym-minecraft](https://github.com/tambetm/gym-minecraft)
* More RL algorithms
  * [DDQN](https://arxiv.org/abs/1509.06461), [Dueling network](https://arxiv.org/abs/1511.06581), [PPO](https://arxiv.org/abs/1707.06347), [QRDQN](https://arxiv.org/abs/1710.10044), [TD3](https://arxiv.org/abs/1802.09477)
  * [Prioritized experience replay](https://arxiv.org/abs/1511.05952), [parameter noise](https://arxiv.org/abs/1706.01905)

## License

Border is primarily distributed under the terms of both the MIT license and the Apache License (Version 2.0).
