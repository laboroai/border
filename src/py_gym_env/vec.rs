#![allow(unused_variables, unreachable_code)]
use std::{fmt::Debug, error::Error};
use std::marker::PhantomData;
use log::{trace};
use pyo3::{PyObject, PyResult, ToPyObject};
use pyo3::types::{PyTuple, IntoPyDict};
use crate::core::{Obs, Act, Step, Env};
use crate::py_gym_env::PyGymInfo;

use super::ObsFilter;

#[derive(Debug, Clone)]
pub struct PyVecGymEnv<O, A, F> {
    env: PyObject,
    n_procs: usize,
    obs_filters: Vec<F>,
    phantom: PhantomData<(O, A, F)>,
}

/// Vectorized version of the gym environment.
/// Adapted from tch-rs RL example.
impl<O, A, F> PyVecGymEnv<O, A, F> where 
    O: Obs,
    A: Act + Into<PyObject>,
    F: ObsFilter<O>,
{
    pub fn new(name: &str, obs_filters: Vec<F>, continuous_action: bool) -> PyResult<Self> {
        let n_procs = obs_filters.len();

        pyo3::Python::with_gil(|py| {
            // sys.argv is used by pyglet library, which is responsible for rendering.
            // Depending on the environment, however, sys.argv can be empty.
            // For that case, sys argv is set here.
            // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
            let locals = [("sys", py.import("sys")?)].into_py_dict(py);
            let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

            let sys = py.import("sys")?;
            let path = sys.get("path")?;
            let _ = path.call_method("append", ("examples",), None)?;
            let gym = py.import("atari_wrappers")?;
            let env = gym.call("make", (name, Option::<&str>::None, n_procs), None)?;

            // let action_space = env.getattr("action_space")?;
            // let action_space = if let Ok(val) = action_space.getattr("n") {
            //     val.extract()?
            // } else {
            //     let action_space: Vec<i64> = action_space.getattr("shape")?.extract()?;
            //     action_space[0]
            // };
            // let observation_space = env.getattr("observation_space")?;
            // let observation_space = observation_space.getattr("shape")?.extract()?;

            Ok(PyVecGymEnv {
                env: env.into(),
                n_procs,
                obs_filters,
                phantom: PhantomData,
            })
        })
    }

    pub fn close(&self) {
        pyo3::Python::with_gil(|py| {
            let _ = self.env.call_method0(py, "close");
        })
    }
}

impl<O, A, F> Env for PyVecGymEnv<O, A, F> where
    O: Obs,
    A: Act + Into<PyObject>,
    F: ObsFilter<O>,
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, returning the observation tensor.
    /// If `is_done` is None, all environemnts are resetted.
    /// If `is_done` is `Vec<f32>`, environments with `is_done[i] == 1.0` are resetted.
    fn reset(&self, is_done: Option<&Vec<f32>>) -> Result<O, Box<dyn Error>>  {
        trace!("PyVecGymEnv::reset()");
        unimplemented!();
        // match is_done {
        //     None => {
        //         pyo3::Python::with_gil(|py| {
        //             let obs = self.env.call_method0(py, "reset")?;
        //             Ok(self.obs_filters[0].reset(obs))
        //         })
        //     },
        //     Some(v) => {
        //         debug_assert_eq!(is_done.len(), self.n_procs);
        //         pyo3::Python::with_gil(|py| {
        //             let obs = self.env.call_method1(py, "reset", is_done)?;
        //            Ok(self.obs_filters[0].reset(obs))
        //         })
        //     }
        // }
    }
    
    fn step(&self, a: &A) -> Step<Self> {
        trace!("{:?}", &a);
        unimplemented!();
        // pyo3::Python::with_gil(|py| {
        //     let a_py = a.clone().into();
        //     let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
        //     let step: &PyTuple = ret.extract(py).unwrap();
        //     let obs = step.get_item(0).to_owned();
        //     let obs = obs.to_object(py).into();
        //     // let reward: f32 = step.get_item(1).extract().unwrap();
        //     // let is_done: bool = step.get_item(2).extract().unwrap();

        //     let reward = Vec::<f32>::new();
        //     let is_done = Vec::<f32>::new();
        //     unimplemented!();

        //     Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo{})
        // })
    }
}
