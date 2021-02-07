#![allow(unused_variables, unreachable_code)]
use std::{fmt::Debug, error::Error};
use std::marker::PhantomData;
use log::{trace};
use pyo3::{
    PyObject, PyResult, ToPyObject,
    types::{IntoPyDict, PyTuple}
};

use crate::{
    core::{Obs, Act, Step, Env},
    env::py_gym_env::{PyGymInfo, PyGymEnvObsFilter, PyGymEnvActFilter}
};

#[derive(Debug, Clone)]
pub struct PyVecGymEnv<O, A, OF, AF> {
    env: PyObject,
    n_procs: usize,
    obs_filter: OF,
    act_filter: AF,
    phantom: PhantomData<(O, A)>,
}

/// Vectorized version of the gym environment.
/// Adapted from tch-rs RL example.
impl<O, A, OF, AF> PyVecGymEnv<O, A, OF, AF> where 
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    pub fn new(name: &str, n_procs: usize, obs_filter: OF, act_filter: AF) -> PyResult<Self> {
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
                obs_filter,
                act_filter,
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

impl<O, A, OF, AF> Env for PyVecGymEnv<O, A, OF, AF> where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, returning the observation tensor.
    ///
    /// If `is_done` is None, all environemnts are resetted.
    /// If `is_done` is `Vec<f32>`, environments with `is_done[i] == 1.0` are resetted.
    fn reset(&mut self, is_done: Option<&Vec<f32>>) -> Result<O, Box<dyn Error>>  {
        trace!("PyVecGymEnv::reset()");

        pyo3::Python::with_gil(|py| {
            let obs = match is_done {
                None => self.env.call_method0(py, "reset").unwrap(),
                Some(v) => self.env.call_method1(py, "reset", (v.clone(),)).unwrap()
            };
            Ok(self.obs_filter.reset(obs))
        })
    }

        //         }
        //     )
        // })
        // match is_done {
        //     None => {
        //         pyo3::Python::with_gil(|py| {
        //             let obs = self.env.call_method0(py, "reset").unwrap();
        //             let obs: Py<PyList> = obs.extract(py).unwrap();
        //             let filtered = self.obs_filters.iter_mut()
        //                 .zip(obs.as_ref(py).iter())
        //                 .map(|(f, o)| f.reset(o.into()))
        //                 .collect();
        //             Ok(OF::stack(filtered))
        //         })
        //     },
        //     Some(v) => {
        //         pyo3::Python::with_gil(|py| {
        //             let obs = self.env.call_method1(py, "reset", (v.clone(),)).unwrap();
        //             let obs: Py<PyList> = obs.extract(py).unwrap();
        //             let filtered = self.obs_filters.iter_mut()
        //                 .zip(obs.as_ref(py).iter())
        //                 .map(|(f, o)| {
        //                     if o.get_type().name().unwrap() == "NoneType" {
        //                         O::zero(1)
        //                     }
        //                     else {
        //                         debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
        //                         f.reset(o.into())
        //                     }
        //                 }).collect();
        //             Ok(OF::stack(filtered))
        //         })
        //     }
        // }
    // }
    
    fn step(&mut self, a: &A) -> Step<Self> {
        trace!("PyVecGymEnv::step()");
        trace!("{:?}", &a);

        pyo3::Python::with_gil(|py| {
            // Does not support render

            let a_py = self.act_filter.filt(a.clone());
            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();
            let obs = step.get_item(0).to_object(py);
            // let obs: Py<PyList> = obs.extract(py).unwrap();
            let obs = self.obs_filter.filt(obs); //iter_mut()
            //     .zip(obs.as_ref(py).iter())
            //     .map(|(f, o)| f.reset(o.into()))
            //     .collect();
            // let obs = OF::stack(filtered);

            // Reward and is_done
            let reward = step.get_item(1).to_object(py);
            let reward: Vec<f32> = reward.extract(py).unwrap();
            let is_done = step.get_item(2).to_object(py);
            let is_done: Vec<f32> = is_done.extract(py).unwrap();

            Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo{})
        })
    }
}
