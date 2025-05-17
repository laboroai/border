#![allow(rustdoc::broken_intra_doc_links)]
//! A wrapper of [Gymnasium](https://gymnasium.farama.org) environments on Python.
//!
//! [`GymEnv`] is a wrapper of [Gymnasium](https://gymnasium.farama.org) based on [`PyO3`](https://github.com/PyO3/pyo3).
//! It has been tested on some of [classic control](https://gymnasium.farama.org/environments/classic_control/) and
//! [Gymnasium-Robotics](https://robotics.farama.org) environments.
//!
//! In order to bridge Python and Rust, we need to convert Python objects to Rust objects and vice versa.
//! This crate provides the [`GymEnvConverter`] trait to handle these conversions.
//!
//! # Type Conversion
//!
//! The [`GymEnvConverter`] trait provides a unified interface for converting between Python and Rust types:
//!
//! * `filt_obs`: Converts Python observations to Rust types
//! * `filt_act`: Converts Rust actions to Python types
//!
//! # Implementations
//!
//! This crate provides several implementations of [`GymEnvConverter`]:
//!
//! * [`ndarray::NdarrayConverter`]: Handles conversions for environments using ndarray types
//! * [`candle::CandleConverter`]: Handles conversions for environments using Candle tensor types
//!   (requires `candle` feature flag)
//! * [`tch::TchConverter`]: Handles conversions for environments using Tch tensor types
//!   (requires `tch` feature flag)
//!
//! To use Candle or Tch converters, enable the corresponding feature in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! border-py-gym-env = { version = "0.1.0", features = ["candle"] }  # For Candle support
//! # or
//! border-py-gym-env = { version = "0.1.0", features = ["tch"] }     # For Tch support
//! ```
//!
//! Each implementation supports different types of observations and actions:
//!
//! * Array observations (e.g., CartPole)
//! * Dictionary observations (e.g., FetchPickAndPlace)
//! * Discrete actions (e.g., CartPole)
//! * Continuous actions (e.g., Pendulum)
//!
//! [`Policy`]: border_core::Policy
//! [`ArrayD`]: https://docs.rs/ndarray/0.15.1/ndarray/type.ArrayD.html
mod base;
#[cfg(feature = "candle")]
pub mod candle;
pub mod ndarray;
#[cfg(feature = "tch")]
pub mod tch;
pub mod util;
pub use base::{GymEnv, GymEnvConfig, GymEnvConverter, GymInfo};
