//! Gym environment in Python.
use std::time::Duration;

use super::{GymActFilter, GymObsFilter};
use crate::AtariWrapper;
use border_core::{Act, Obs};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
/// Configuration of [`GymEnv`](super::GymEnv).
pub struct GymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    /// The maximum interaction steps in an episode.
    pub max_steps: Option<usize>,

    /// If not `None`, a function in `examples/atari_wrappers.py` is called and
    /// environment wrappers for Atari will be applied.
    /// Otherwise, a function in `f32_wrappers.py` is called to make an environment (not Atari).
    pub atari_wrapper: Option<AtariWrapper>,

    /// `true` to support rendering for PyBullet gym environment.
    pub pybullet: bool,

    /// Name of the environment, e.g., `CartPole-v0`.
    pub name: String,

    /// Rendering mode, e.g., "human" or "rgb_array".
    pub render_mode: Option<String>,

    /// Configuration of [`GymObsFilter`].
    pub obs_filter_config: Option<OF::Config>,

    /// Configuration of [`GymActFilter`].
    pub act_filter_config: Option<AF::Config>,

    /// Wait time at every interaction steps.
    pub wait: Duration
}

impl<O, A, OF, AF> Clone for GymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    fn clone(&self) -> Self {
        Self {
            max_steps: self.max_steps,
            atari_wrapper: self.atari_wrapper.clone(),
            pybullet: self.pybullet,
            name: self.name.clone(),
            render_mode: self.render_mode.clone(),
            obs_filter_config: self.obs_filter_config.clone(),
            act_filter_config: self.act_filter_config.clone(),
            wait: self.wait.clone(),
        }
    }
}

impl<O, A, OF, AF> Default for GymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    fn default() -> Self {
        Self {
            max_steps: None,
            atari_wrapper: None,
            pybullet: false,
            name: "".to_string(),
            render_mode: None,
            obs_filter_config: None,
            act_filter_config: None,
            wait: Duration::from_millis(0),
        }
    }
}

impl<O, A, OF, AF> GymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
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

    pub fn render_mode(mut self, render_mode: Option<String>) -> Self {
        self.render_mode = render_mode;
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

    /// Set wait time in milli seconds.
    pub fn set_wait_in_millis(mut self, millis: u64) -> Self {
        self.wait = Duration::from_millis(millis);
        self
    }
}
