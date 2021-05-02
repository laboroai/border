//! Vectorized environment using multiprocess module in Python.
#![allow(unused_variables, unreachable_code)]
use log::trace;
use pyo3::{
    types::{IntoPyDict, PyTuple},
    PyObject, PyResult, ToPyObject,
};
use std::marker::PhantomData;
use std::{error::Error, fmt::Debug};

use super::AtariWrapper;
use crate::{
    core::{record::Record, Act, Env, Obs, Step},
    env::py_gym_env::{PyGymEnvActFilter, PyGymEnvObsFilter, PyGymInfo},
};

/// Constructs [PyVecGymEnv]
pub struct PyVecGymEnvBuilder<O, A, OF, AF>  {
    atari_wrapper: Option<AtariWrapper>,
    n_procs: usize,
    phantom: PhantomData<(O, A, OF, AF)>,
}

impl <O, A, OF, AF> Default for PyVecGymEnvBuilder<O, A, OF, AF> {
    fn default() -> Self {
        Self {
            atari_wrapper: None,
            n_procs: 1,
            phantom: PhantomData,
        }
    }
}

impl<O, A, OF, AF> PyVecGymEnvBuilder<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{

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

    /// Constructs [PyVecGymEnv].
    ///
    /// * `name` - Name of a gym environment.
    pub fn build(
        self,
        name: &str,
        obs_filter: OF,
        act_filter: AF,
    ) -> PyResult<PyVecGymEnv<O, A, OF, AF>> {
        pyo3::Python::with_gil(|py| {
            // sys.argv is used by pyglet library, which is responsible for rendering.
            // Depending on the python interpreter, however, sys.argv can be empty.
            // For that case, sys argv is set here.
            // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
            let locals = [("sys", py.import("sys")?)].into_py_dict(py);
            let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

            let env = if let Some(mode) = self.atari_wrapper {
                let mode = match mode {
                    AtariWrapper::Train => true,
                    AtariWrapper::Eval => false,
                };
                let gym = py.import("atari_wrappers")?;
                let env = gym.call("make", (name, true, mode, self.n_procs), None)?;
                env
            } else {
                unimplemented!();
                // let gym = py.import("gym")?;
                // let env = gym.call("make", (name,), None)?;
                // env.call_method("seed", (42,), None)?;
                // env
            };

            Ok(PyVecGymEnv {
                env: env.into(),
                n_procs: self.n_procs,
                obs_filter,
                act_filter,
                phantom: PhantomData,
            })
        })
    }
}
    

/// A vectorized environment using multiprocess module in Python.
/// The code is adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning).
#[derive(Debug, Clone)]
pub struct PyVecGymEnv<O, A, OF, AF> {
    env: PyObject,
    n_procs: usize,
    obs_filter: OF,
    act_filter: AF,
    phantom: PhantomData<(O, A)>,
}

impl<O, A, OF, AF> PyVecGymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    // /// Constructs a vectorized environment.
    // ///
    // /// This function invl
    // ///
    // /// * `atari_wrapper` - If `true`, `wrap_deepmind()` is applied to the environment.
    // ///   See `atari_wrapper.py`.
    // pub fn new(
    //     name: &str,
    //     n_procs: usize,
    //     obs_filter: OF,
    //     act_filter: AF,
    //     atari_wrapper: bool,
    // ) -> PyResult<Self> {
    //     pyo3::Python::with_gil(|py| {
    //         // sys.argv is used by pyglet library, which is responsible for rendering.
    //         // Depending on the python interpreter, however, sys.argv can be empty.
    //         // For that case, sys argv is set here.
    //         // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
    //         let locals = [("sys", py.import("sys")?)].into_py_dict(py);
    //         let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

    //         // TODO: Consider removing addition of pythonpath
    //         let sys = py.import("sys")?;
    //         let path = sys.get("path")?;
    //         let _ = path.call_method("append", ("examples",), None)?;

    //         let gym = py.import("atari_wrappers")?;
    //         let env = gym.call(
    //             "make",
    //             (name, Option::<&str>::None, atari_wrapper, n_procs),
    //             None,
    //         )?;

    //         Ok(PyVecGymEnv {
    //             env: env.into(),
    //             n_procs,
    //             obs_filter,
    //             act_filter,
    //             phantom: PhantomData,
    //         })
    //     })
    // }

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
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, the obs/act filters and returns the observation tensor.
    ///
    /// If `is_done` is None, all environemnts are resetted.
    /// Otherwise, `is_done` is `Vec<f32>` and environments with `is_done[i] == 1.0` are resetted.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<O, Box<dyn Error>> {
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

            let step = Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo {});
            let record = record_o.merge(record_a);

            (step, record)
        })
    }
}
