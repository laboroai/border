//! Vectorized environment using multiprocess module in Python.
#![allow(unused_variables, unreachable_code)]
use super::PyVecGymEnvConfig;
use crate::AtariWrapper;
use crate::{GymActFilter, GymInfo, GymObsFilter};
use anyhow::Result;
use border_core::{record::Record, Act, Env, Obs, Step};
use log::trace;
use pyo3::{
    types::{IntoPyDict, PyTuple},
    PyObject, ToPyObject,
};
use std::{fmt::Debug, marker::PhantomData};

/// A vectorized environment using multiprocess module in Python.
/// The code is adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning).
#[derive(Debug, Clone)]
pub struct PyVecGymEnv<O, A, OF, AF> {
    env: PyObject,
    max_steps: Option<usize>,
    n_procs: usize,
    obs_filter: OF,
    act_filter: AF,
    phantom: PhantomData<(O, A)>,
}

impl<O, A, OF, AF> PyVecGymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    /// Get the number of available actions of atari environments
    pub fn get_num_actions_atari(&self) -> i64 {
        pyo3::Python::with_gil(|py| {
            let act_space = self.env.getattr(py, "action_space").unwrap();
            act_space.getattr(py, "n").unwrap().extract(py).unwrap()
        })
    }

    /// Close all subprocesses.
    ///
    /// TODO: Consider implementing the method in `Drop` trait.
    pub fn close(&self) {
        pyo3::Python::with_gil(|py| {
            let _ = self.env.call_method0(py, "close");
        })
    }
}

impl<O, A, OF, AF> Env for PyVecGymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    type Obs = O;
    type Act = A;
    type Info = GymInfo;
    type Config = PyVecGymEnvConfig<O, A, OF, AF>;

    /// Constructs [PyVecGymEnv].
    ///
    /// * `name` - Name of a gym environment.
    fn build(config: &Self::Config, seed: i64) -> Result<Self> {
        pyo3::Python::with_gil(|py| {
            // sys.argv is used by pyglet library, which is responsible for rendering.
            // Depending on the python interpreter, however, sys.argv can be empty.
            // For that case, sys argv is set here.
            // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
            let locals = [("sys", py.import("sys")?)].into_py_dict(py);
            let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

            let gym = py.import("atari_wrappers")?;
            let name = config.name.as_str();
            let env = if let Some(mode) = config.atari_wrapper.as_ref() {
                let mode = match mode {
                    AtariWrapper::Train => true,
                    AtariWrapper::Eval => false,
                };
                // gym.call("make", (name, true, mode, config.n_procs), None)?
                gym.getattr("make")?
                    .call((name, true, mode, config.n_procs), None)?
            } else {
                // gym.call("make", (name, false, false, config.n_procs), None)?
                gym.getattr("make")?
                    .call((name, false, false, config.n_procs), None)?
            };

            Ok(PyVecGymEnv {
                max_steps: config.max_steps,
                env: env.into(),
                n_procs: config.n_procs,
                obs_filter: OF::build(config.obs_filter_config.as_ref().unwrap())?,
                act_filter: AF::build(config.act_filter_config.as_ref().unwrap())?,
                phantom: PhantomData,
            })
        })
    }

    fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized,
    {
        unimplemented!();
    }

    /// Resets the environment, the obs/act filters and returns the observation tensor.
    ///
    /// If `is_done` is None, all environemnts are resetted.
    /// Otherwise, `is_done` is `Vec<f32>` and environments with `is_done[i] == 1.0` are resetted.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<O> {
        trace!("PyVecGymEnv::reset()");

        // Reset the action filter, required for stateful filters.
        self.act_filter.reset(&is_done);

        pyo3::Python::with_gil(|py| {
            let obs = match is_done {
                None => self.env.call_method0(py, "reset").unwrap(),
                Some(v) => self.env.call_method1(py, "reset", (v.clone(),)).unwrap(),
            };
            Ok(self.obs_filter.reset(obs))
        })
    }

    fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs> {
        unimplemented!();
    }

    fn step(&mut self, a: &A) -> (Step<Self>, Record) {
        trace!("PyVecGymEnv::step()");
        trace!("{:?}", &a);

        pyo3::Python::with_gil(|py| {
            // Does not support render

            let (a_py, record_a) = self.act_filter.filt(a.clone());
            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();
            let obs = step.get_item(0).to_object(py);
            let (obs, record_o) = self.obs_filter.filt(obs);

            // Reward and is_done
            let reward = step.get_item(1).to_object(py);
            let reward: Vec<f32> = reward.extract(py).unwrap();
            let is_done = step.get_item(2).to_object(py);
            let is_done: Vec<f32> = is_done.extract(py).unwrap();
            let is_done: Vec<i8> = is_done.into_iter().map(|x| x as i8).collect();
            let n = obs.len();

            let step = Step::<Self>::new(obs, a.clone(), reward, is_done, GymInfo {}, O::dummy(n));
            let record = record_o.merge(record_a);

            (step, record)
        })
    }
}
