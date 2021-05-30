#![warn(missing_docs)]
//! Border is a library for reinforcement learning (RL). The aim of the library is to provide
//! both environment- and agent-agnostic framework based on the expressiveness and type-safety
//! of Rust. In order to do this, border provides a function for evaluation ([eval]) and
//! a trait for training ([Trainer]). These function and trait use some traits consisting
//! main components in RL such as observation ([Obs]), action ([Act]), and environment ([Env]).
//! It should be noted that in border, [Policy] represents controllers, which interacts with
//! environments, while [Agent] provides the interface to train policies by working on [Trainer].
//!
//! Using these components, border implements build-in environments and RL algorithms.
//! [PyGymEnv] and [PyVecGymEnv] are wrapper around [OpenAI gym](https://gym.openai.com)
//! implemented in Python. These wrappers allow the library to use a lot of environments,
//! including [atari 2600](https://gym.openai.com/envs/#atari) and
//! [pybullet-gym](https://github.com/benelot/pybullet-gym), commonly used in the RL comminity.
//!
//! Border has some RL agents for discrete action ([DQN]) and continuous action ([SAC]).
//! These agents use [tch], a wrapper of libtorch.
//!
//! Border enables to make environments and agents be independend regarding their internal
//! computation. Actually, [PyGymEnv] and [PyVecGymEnv] use [ndarray] in order to represent observation and action.
//! In contrast, RL agents mentioned above use [tch::Tensor] for internal computation.
//! Functions and traits in [env::py_gym_env::tch] bridges the gap between these environments and agents.
//! Like this, border is flexible in combination of environments and agents.
//! By providing some bridges, it is able to switch computational backends or internal implementation of
//! environments and agents. This concept is illustrated in schematic diagram of [eval] and [Trainer].
pub mod agent;
pub mod env;
pub mod error;
pub mod util;

#[allow(unused_imports)]
use {
    crate::{
        agent::tch::{DQN, SAC},
        env::py_gym_env::{PyGymEnv, PyVecGymEnv},
    },
    border_core::{eval, Act, Agent, Env, Obs, Policy, Trainer},
};
