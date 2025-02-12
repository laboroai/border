//! A wrapper of [Gymnasium](https://gymnasium.farama.org) environments on Python.
//!
//! [`GymEnv`] is a wrapper of [Gymnasium](https://gymnasium.farama.org) based on [`PyO3`](https://github.com/PyO3/pyo3).
//! It has been tested on some of [classic control](https://gymnasium.farama.org/environments/classic_control/) and
//! [Gymnasium-Robotics](https://robotics.farama.org) environments.
//!
//! In order to bridge Python and Rust, we need to convert Python objects to Rust objects and vice versa.
//!
//! ## Observation
//!
//! Observation is created in Python and passed to Rust as a Python object. In order to convert
//! Python object to Rust object, this crate provides [`GymObsFilter`] trait. This trait has
//! [`GymObsFilter::filt`] method which converts Python object to Rust object.
//! The type of the Rust object after conversion corresponds to the type parameter `O` of the trait
//! and this is also the type of the observation in the environment, i.e., [`GymEnv`]`::Obs`.
//!
//! There are two built-in implementations of [`GymObsFilter`]: [`ArrayObsFilter`] and [`ArrayDictObsFilter`].
//! [`ArrayObsFilter`] is for environments where observation is an array (e.g., CartPole).
//! Internally, the array is converted to [`ndarray::ArrayD`] from Python object.
//! Then, the array is converted to the type parameter `O` of the filter.
//! Since `O` must implement [`From<ndarray::ArrayD>`] by trait bound, the conversion is done
//! by calling `array.into()`.
//!
//! [`ArrayDictObsFilter`] is for environments where observation is a dictionary of arrays (e.g., FetchPickAndPlace).
//! Internally, the dictionary is converted to `Vec<(String, border_py_gym_env:util::Array)>` from Python object.
//! Then, `Vec<(String, border_py_gym_env:util::Array)>` is converted to `O` by calling `into()`.
//!
//! ## Action
//!
//! Action is created in [`Policy`] and passed to Python as a Python object. In order to convert
//! Rust object to Python object, this crate provides [`GymActFilter`] trait. This trait has
//! [`GymActFilter::filt`] method which converts Rust object of type `A`, which is the type parameter of
//! the trait, to Python object.
//!
//! There are two built-in implementations of [`GymActFilter`]: [`DiscreteActFilter`] and [`ContinuousActFilter`].
//! [`DiscreteActFilter`] is for environments where action is discrete (e.g., CartPole).
//! This filter converts `A` to [`Vec<i32>`] and then to Python object.
//! [`ContinuousActFilter`] is for environments where action is continuous (e.g., Pendulum).
//! This filter converts `A` to [`ArrayD`] and then to Python object.
//!
//! [`Policy`]: border_core::Policy
//! [`ArrayD`]: https://docs.rs/ndarray/0.15.1/ndarray/type.ArrayD.html
// mod act;
mod base;
#[cfg(feature = "candle")]
pub mod candle;
// mod config;
// mod obs;
pub mod util;
// pub use act::{
//     ContinuousActFilter, ContinuousActFilterConfig, DiscreteActFilter, DiscreteActFilterConfig,
// };
pub use base::{GymEnv, GymEnvConfig, GymEnvConverter, GymInfo};
// #[allow(deprecated)]
// pub use obs::{ArrayDictObsFilter, ArrayDictObsFilterConfig, ArrayObsFilter, ArrayObsFilterConfig};
