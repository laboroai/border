//! Vectorized environment using multiprocess module in Python.
#![allow(unused_variables, unreachable_code)]
use crate::AtariWrapper;
use crate::{GymActFilter, GymObsFilter};
use border_core::{Act, Obs};
use std::marker::PhantomData;

/// Constructs PyVecGymEnv
pub struct PyVecGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    // Name of the environment
    pub name: String,
    pub max_steps: Option<usize>,
    pub atari_wrapper: Option<AtariWrapper>,
    // The number of processes
    pub n_procs: usize,
    pub obs_filter_config: Option<OF::Config>,
    pub act_filter_config: Option<AF::Config>,
    phantom: PhantomData<(O, A, OF, AF)>,
}

impl<O, A, OF, AF> Clone for PyVecGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            max_steps: self.max_steps,
            atari_wrapper: self.atari_wrapper.clone(),
            n_procs: self.n_procs,
            obs_filter_config: self.obs_filter_config.clone(),
            act_filter_config: self.act_filter_config.clone(),
            phantom: PhantomData,
        }
    }
}

impl<O, A, OF, AF> Default for PyVecGymEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
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
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
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
