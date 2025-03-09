//! Interface to access [Minari](https://minari.farama.org/index.html#) datasets.
//!
//! [`MinariDataset`] and [`MinariEnv`] provide a common interface to access Minari datasets.
//! These structs are used with concrete observation and action types.
//! For example, [`border_minari::d4rl::kitchen`] provides observation and action types,
//! and the corresponding converter for the [Kitchen datasets](https://minari.farama.org/datasets/D4RL/kitchen/).
//!
//! The implementation of data types and converters depends on the backend for implementing
//! your agents. In [`border_minari::d4rl::kitchen::ndarray`], the observation and action types are
//! defined essentially as [`ndarray::ArrayD`]. In [`border_minari::d4rl::kitchen::candle`],
//! the observation and action types are defined as [`candle_core::Tensor`].
//!
//! In the below example, we load an episode in the Kitchen dataset and create a replay buffer for that.
//! Then, we recover the environment from the dataset and apply the actions in the episode.
//! The observation and action types are implemented with [`ndarray::ArrayD`].
//!
//! ```no_run
//! # use anyhow::Result;
//! use border_core::Env;
//! use border_minari::{d4rl::kitchen::ndarray::KitchenConverter, MinariDataset};
//! # use numpy::convert;
//! # use std::num;
//!
//! fn main() -> Result<()> {
//!     let dataset = MinariDataset::load_dataset("D4RL/kitchen/complete-v1", true)?;
//!
//!     // Converter for observation and action
//!     let converter = KitchenConverter {};
//!
//!     // Create replay buffer for the sixth episode
//!     let replay_buffer = dataset.create_replay_buffer(&converter, Some(vec![5]))?;
//!
//!     // Recover the environment from the dataset
//!     let mut env = dataset.recover_environment(converter, false, "human")?;
//!
//!     // Sequence of actions in the episode
//!     let actions = replay_buffer.whole_actions();
//!
//!     // Apply the actions to the environment
//!     env.reset(None)?;
//!     for ix in 0..actions.action.shape()[0] {
//!         let act = actions.get(ix);
//!         let _ = env.step(&act);
//!     }
//!
//!     Ok(())
//! }
//! ```
//! [`candle_core::Tensor`]: candle_core::Tensor
//! [`border_minari::d4rl::kitchen`]: crate::d4rl::kitchen
//! [`border_minari::d4rl::kitchen::ndarray`]: crate::d4rl::kitchen::ndarray
//! [`border_minari::d4rl::kitchen::candle`]: crate::d4rl::kitchen::candle
mod converter;
mod dataset;
mod env;
mod evaluator;

pub mod d4rl;
pub mod util;

pub use converter::MinariConverter;
pub use dataset::MinariDataset;
pub use env::MinariEnv;
pub use evaluator::MinariEvaluator;
