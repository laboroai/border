//! Vectorized environment using multiprocess module in Python.
#![allow(unused_variables, unreachable_code)]
use crate::AtariWrapper;
use crate::{PyGymEnvActFilter, PyGymEnvObsFilter};
use border_core::{Act, Obs};
use std::marker::PhantomData;

/// Constructs [PyVecGymEnv](super::PyVecGymEnv)
pub struct PyVecGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    // Name of the environment
    pub(super) name: String,
    pub(super) max_steps: Option<usize>,
    pub(super) atari_wrapper: Option<AtariWrapper>,
    // The number of processes
    pub(super) n_procs: usize,
    pub(super) obs_filter_config: Option<OF::Config>,
    pub(super) act_filter_config: Option<AF::Config>,
    phantom: PhantomData<(O, A, OF, AF)>,
}

impl<O, A, OF, AF> Default for PyVecGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    fn default() -> Self {
        Self {
            name: "".to_string(),
            max_steps: None,
            atari_wrapper: None,
            n_procs: 1,
            obs_filter_config: None,
            act_filter_config: None,
            phantom: PhantomData,
        }
    }
}

impl<O, A, OF, AF> PyVecGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    /// Sets the maximum number of steps in the environment.
    pub fn max_steps(mut self, max_steps: Option<usize>) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets `True` when using Atari wrapper.
    pub fn atari_wrapper(mut self, v: Option<AtariWrapper>) -> Self {
        self.atari_wrapper = v;
        self
    }

    /// Sets the number of processes.
    pub fn n_procs(mut self, v: usize) -> Self {
        self.n_procs = v;
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
