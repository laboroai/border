# TODO

* Image observation
* Logging
  * TD-error
* Composition of exploration strategy
* Vectorized environment
* Documentation
  * Default values of training/algorithm parameters
* Add/improve RL methods
  * DDQN
  * SAC
    * Double Q
    * EntCoef auto tuning
  * DDPG
    * Fix adhoc implementation of scaling for pendulum env
      * (See sac_pendulum.rs for action scaling)
  * Prioritized experience replay
  * Parameter noise
* GPU support
  * Benchmarking
* Add more examples
  * PyBullet gym
  * Atari

# Examples

TODO: consider adding following examples as test

TODO: organize examples in `example/` directory, env-first

* Cartpole
  * random_cartpole
  * dqn_cartpole
* Pendulum
  * ddpg_pendulum
  * sac_pendulum
* LunarLander-cont
  * random_lunarlander_cont
  * ddpg_lunarlander_cont
  * sac_lunarlander_cont
* Cartpole, vectorized
  * random_cartpole_vecenv_test.rs // TODO: consider removing `_test`
  * dqn_cartpole_vecenv
* LunarLander-cont, vectorized
  * sac_lunarlander_cont_vecenv
* Pong
  * dqn_pong

# References

## Alaorithms

* DQN
* DoubleDQN
* SAC
* DDPG
* PPO
* QRDQN
* PER
  * [Let’s make a DQN: Double Learning and Prioritized Experience Replay](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/)
  * [Understanding Prioritized Experience Replay](https://danieltakeshi.github.io/2019/07/14/per/)

## Environments

* [Table of Environments in gym](https://github.com/openai/gym/wiki/Table-of-environments)
* [pybullet-gym](https://github.com/benelot/pybullet-gym)
* [ViZDoom](https://github.com/mwydmuch/ViZDoom)

## Benchmark

* [Reproduction of Atari in PFRL](https://github.com/pfnet/pfrl/tree/master/examples/atari/reproduction/dqn)

## RL implementation

* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [tensorflow-rl-pong](https://github.com/mrahtz/tensorflow-rl-pong)

## Rust crates

* [tch-rs](https://crates.io/crates/tch)
* [ndarray](https://crates.io/crates/ndarray)
* [pyo3](https://crates.io/crates/pyo3)
* [rust-numpy](https://crates.io/crates/numpy)
* [tensorboard-rs](https://crates.io/crates/tensorboard-rs)

## Development

* [The Cargo Book](https://doc.rust-lang.org/cargo/index.html#the-cargo-book)
  * [Publishing on crates.io](https://doc.rust-lang.org/cargo/reference/publishing.html)
    * [The version field](https://doc.rust-lang.org/cargo/reference/manifest.html#the-version-field)
  * [Specifying Dependencies](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)
    * [Development dependencies](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#development-dependencies)
* [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/about.html)
* [Rust Cookbook](https://rust-lang-nursery.github.io/rust-cookbook/intro.html)
  * [Argument Parsing](https://rust-lang-nursery.github.io/rust-cookbook/cli/arguments.html)
