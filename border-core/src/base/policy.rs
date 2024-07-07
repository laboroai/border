//! Policy.
use super::Env;
use anyhow::Result;
use serde::de::DeserializeOwned;
use std::path::Path;

/// A policy on an environment.
///
/// Policy is a mapping from an observation to an action.
/// The mapping can be either of deterministic or stochastic.
pub trait Policy<E: Env> {
    /// Sample an action given an observation.
    fn sample(&mut self, obs: &E::Obs) -> E::Act;
}

/// A configurable object, having type parameter.
pub trait Configurable<E: Env> {
    /// Configuration.
    type Config: Clone + DeserializeOwned;

    /// Builds the object.
    fn build(config: Self::Config) -> Self;

    /// Build the object with the configuration in the yaml file of the given path.
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
