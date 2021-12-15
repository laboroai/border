//! Gym environment in Python.
use super::{PyGymEnvActFilter, PyGymEnvObsFilter};
use crate::AtariWrapper;
use border_core::{Act, Obs};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
/// Configuration of [PyGymEnv](super::PyGymEnv).
pub struct PyGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    pub(super) max_steps: Option<usize>,
    pub(super) atari_wrapper: Option<AtariWrapper>,
    pub(super) pybullet: bool,
    pub(super) name: String,
    pub(super) obs_filter_config: Option<OF::Config>,
    pub(super) act_filter_config: Option<AF::Config>,
}

impl<O, A, OF, AF> Clone for PyGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    fn clone(&self) -> Self {
        Self {
            max_steps: self.max_steps,
            atari_wrapper: self.atari_wrapper.clone(),
            pybullet: self.pybullet,
            name: self.name.clone(),
            obs_filter_config: self.obs_filter_config.clone(),
            act_filter_config: self.act_filter_config.clone(),
        }
    }
}

impl<O, A, OF, AF> Default for PyGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    fn default() -> Self {
        Self {
            max_steps: None,
            atari_wrapper: None,
            pybullet: false,
            name: "".to_string(),
            obs_filter_config: None,
            act_filter_config: None,
        }
    }
}

impl<O, A, OF, AF> PyGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    /// Set `True` when using PyBullet environments.
    pub fn pybullet(mut self, v: bool) -> Self {
        self.pybullet = v;
        self
    }

    /// Set `True` when using Atari wrapper.
    pub fn atari_wrapper(mut self, v: Option<AtariWrapper>) -> Self {
        self.atari_wrapper = v;
        self
    }

    /// Set the name of the environment.
    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Set the observation filter config.
    pub fn obs_filter_config(mut self, obs_filter_config: OF::Config) -> Self {
        self.obs_filter_config = Some(obs_filter_config);
        self
    }

    /// Set the action filter config.
    pub fn act_filter_config(mut self, act_filter_config: AF::Config) -> Self {
        self.act_filter_config = Some(act_filter_config);
        self
    }
}
