//! Policy interface for reinforcement learning.
//!
//! This module defines the core interface for policies in reinforcement learning.
//! A policy represents a decision-making strategy that maps observations to actions,
//! which can be either deterministic or stochastic.

use super::Env;
use anyhow::Result;
use serde::de::DeserializeOwned;
use std::path::Path;

/// A policy that maps observations to actions in a reinforcement learning environment.
///
/// This trait defines the interface for policies, which are the core decision-making
/// components in reinforcement learning. A policy can be:
/// - Deterministic: Always returns the same action for a given observation
/// - Stochastic: Returns actions sampled from a probability distribution
///
/// # Type Parameters
///
/// * `E` - The environment type that this policy operates on
///
/// # Examples
///
/// A simple deterministic policy might look like:
/// ```ignore
/// struct SimplePolicy;
///
/// impl<E: Env> Policy<E> for SimplePolicy {
///     fn sample(&mut self, obs: &E::Obs) -> E::Act {
///         // Always return the same action for a given observation
///         E::Act::default()
///     }
/// }
/// ```
///
/// A stochastic policy might look like:
/// ```ignore
/// struct StochasticPolicy;
///
/// impl<E: Env> Policy<E> for StochasticPolicy {
///     fn sample(&mut self, obs: &E::Obs) -> E::Act {
///         // Sample an action from a probability distribution
///         // based on the observation
///         E::Act::random()
///     }
/// }
/// ```
pub trait Policy<E: Env> {
    /// Samples an action given an observation from the environment.
    ///
    /// This method is the core of the policy interface, defining how the policy
    /// makes decisions based on the current state of the environment.
    ///
    /// # Arguments
    ///
    /// * `obs` - The current observation from the environment
    ///
    /// # Returns
    ///
    /// An action to be taken in the environment
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}

/// A trait for objects that can be configured and built from configuration files.
///
/// This trait provides a standardized way to create objects from configuration
/// parameters, either directly or from YAML files. It is commonly used for
/// creating policies, environments, and other components of a reinforcement
/// learning system.
///
/// # Associated Types
///
/// * `Config` - The configuration type that can be deserialized from YAML
///
/// # Examples
///
/// ```ignore
/// #[derive(Clone, Deserialize)]
/// struct MyConfig {
///     learning_rate: f32,
///     hidden_size: usize,
/// }
///
/// struct MyObject {
///     config: MyConfig,
/// }
///
/// impl Configurable for MyObject {
///     type Config = MyConfig;
///
///     fn build(config: Self::Config) -> Self {
///         Self { config }
///     }
/// }
///
/// // Build from a YAML file
/// let obj = MyObject::build_from_path("config.yaml")?;
/// ```
pub trait Configurable {
    /// The configuration type for this object.
    ///
    /// This type must implement `Clone` and `DeserializeOwned` to support
    /// building from configuration files.
    type Config: Clone + DeserializeOwned;

    /// Builds a new instance of this object from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration parameters
    ///
    /// # Returns
    ///
    /// A new instance of the object
    fn build(config: Self::Config) -> Self;

    /// Builds a new instance from a YAML configuration file.
    ///
    /// This is a convenience method that reads a YAML file and builds
    /// the object using the deserialized configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Returns
    ///
    /// A new instance of the object or an error if the file cannot be read
    /// or parsed
    fn build_from_path(path: impl AsRef<Path>) -> Result<Self>
    where
        Self: Sized,
    {
        let file = std::fs::File::open(path)?;
        let rdr = std::io::BufReader::new(file);
        let config = serde_yaml::from_reader(rdr)?;
        Ok(Self::build(config))
    }
}
