//! A wrapper for [Minari](https://minari.farama.org) environments.
//!
//! This crate provides a Rust interface for Minari datasets, which are collections of offline reinforcement learning data.
//! It allows users to load and interact with Minari datasets in a way that is compatible with the Border framework.
//!
//! # Features
//!
//! - **Dataset Loading**: Load Minari datasets from disk or from the Minari registry.
//! - **Environment Interaction**: Interact with the loaded datasets using the Border environment interface.
//! - **Data Access**: Access observations, actions, rewards, and other data from the datasets.
//!
//! # Example
//!
//! The following example demonstrates how to:
//! 1. Load a D4RL Kitchen dataset
//! 2. Create a replay buffer from a specific episode
//! 3. Recover the environment state
//! 4. Replay the actions from the dataset
//!
//! This is particularly useful for:
//! - Analyzing expert demonstrations
//! - Testing environment behavior
//! - Validating dataset quality
//! - Reproducing recorded trajectories
//!
//! ```no_run
//! # use anyhow::Result;
//! use border_core::Env;
//! use border_minari::{d4rl::kitchen::ndarray::KitchenConverter, MinariDataset};
//! # use numpy::convert;
//! # use std::num;
//!
//! fn main() -> Result<()> {
//!     // Load the D4RL Kitchen dataset
//!     let dataset = MinariDataset::load_dataset("D4RL/kitchen/complete-v1", true)?;
//!
//!     // Create a converter for handling observation and action types
//!     let mut converter = KitchenConverter {};
//!
//!     // Create a replay buffer containing only the sixth episode
//!     let replay_buffer = dataset.create_replay_buffer(&mut converter, Some(vec![5]))?;
//!
//!     // Recover the environment state from the dataset
//!     // The 'false' parameter indicates not to use the initial state
//!     // 'human' indicates the agent type
//!     let mut env = dataset.recover_environment(converter, false, "human")?;
//!
//!     // Get the sequence of actions from the replay buffer
//!     let actions = replay_buffer.whole_actions();
//!
//!     // Reset the environment and replay the actions
//!     env.reset(None)?;
//!     for ix in 0..actions.action.shape()[0] {
//!         let act = actions.get(ix);
//!         let _ = env.step(&act);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! The example uses the following key components:
//! - [`KitchenConverter`]: Handles conversion between Python and Rust types for the Kitchen environment
//! - [`MinariDataset`]: Manages the dataset and provides methods for data access
//! - [`Env`]: The Border environment interface for interaction
//!
//! [`KitchenConverter`]: crate::d4rl::kitchen::ndarray::KitchenConverter
//! [`MinariDataset`]: crate::MinariDataset
//! [`Env`]: border_core::Env
//!
//! # Integration with Border
//!
//! This crate implements the [`Env`] trait from `border-core`, making it compatible with other Border components
//! such as agents, policies, and trainers. It can be used in both online and offline reinforcement learning scenarios.
//!
//! [`Env`]: border_core::Env

mod converter;
pub mod d4rl;
mod dataset;
pub mod env;
pub mod evaluator;
pub mod util;
pub use converter::MinariConverter;
pub use dataset::MinariDataset;
pub use env::MinariEnv;
pub use evaluator::MinariEvaluator;
