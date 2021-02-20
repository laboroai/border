#![allow(clippy::float_cmp)]
use std::{fmt::Debug, error::Error};
use std::marker::PhantomData;
use std::cell::RefCell;
use log::{trace};
use pyo3::{PyObject, PyResult, Python, ToPyObject};
use pyo3::types::{PyTuple, IntoPyDict};

use crate::core::{Act, Env, Info, Obs, Step, record::Record};

pub struct PyGymInfo {}

impl Info for PyGymInfo {}

// TODO: consider moving it under src/env/py_gym_env
pub trait Shape: Clone + Debug {
    fn shape() -> &'static [usize];

    /// Return true if you would like to squeeze the first dimension of the array
    /// before conversion into an numpy array in Python. The first dimension may
    /// correspond to process indices for vectorized environments. However, this
    /// dimension is not compatible with PyGymEnv (non-vectorized environment).
    /// This method is used in
    /// [here][crate::agents::tch::py_gym_env::act_c::TchPyGymEnvContinuousAct#impl-Into<Py<PyAny>>].
    fn squeeze_first_dim() -> bool {
        false
    }
}

/// Convert PyObject to PyGymEnv::Obs.
pub trait PyGymEnvObsFilter<O: Obs> {
    /// Convert PyObject into observation with filtering.
    fn filt(&mut self, obs: PyObject) -> (O, Record);

    /// Called when resetting the environment.
    ///
    /// This method is useful for stateful filters.
    fn reset(&mut self, obs: PyObject) -> O {
        let (obs, _) = self.filt(obs);
        obs
    }
}

/// Convert PyGymEnv::Act to PyObject.
///
/// This trait should support vectorized environments.
pub trait PyGymEnvActFilter<A: Act> {
    /// Filter action and convert it to PyObject.
    ///
    /// For vectorized environments, `act` should have actions for all environments in
    /// the vectorized environment. The return values will be a `PyList` object, each
    /// element is an action of the corresponding environment.
    fn filt(&mut self, act: A) -> (PyObject, Record);

    /// Called when resetting the environment.
    ///
    /// This method is useful for stateful filters.
    /// This method support vectorized environment
    fn reset(&mut self, _is_done: &Option<&Vec<f32>>) {}
}

/// Adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning).
/// It represents non-vectorized environment (`n_procs`=1).
#[derive(Debug, Clone)]
pub struct PyGymEnv<O, A, OF, AF> where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>
{
    render: bool,
    env: PyObject,
    action_space: i64,
    observation_space: Vec<usize>,
    count_steps: RefCell<usize>,
    max_steps: Option<usize>,
    obs_filter: OF,
    act_filter: AF,
    phantom: PhantomData<(O, A)>,
}

impl<O, A, OF, AF> PyGymEnv<O, A, OF, AF> where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>
{
    pub fn new(name: &str, obs_filter: OF, act_filter: AF) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // sys.argv is used by pyglet library, which is responsible for rendering.
        // Depending on the environment, however, sys.argv can be empty.
        // For that case, sys argv is set here.
        // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
        let locals = [("sys", py.import("sys")?)].into_py_dict(py);
        let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

        let gym = py.import("gym")?;
        let env = gym.call("make", (name,), None)?;
        let _ = env.call_method("seed", (42,), None)?;

        // TODO: consider removing action_space and observation_space.
        // Act/obs types are specified by type parameters.
        let action_space = env.getattr("action_space")?;
        let action_space = if let Ok(val) = action_space.getattr("n") {
            val.extract()?
        } else {
            let action_space: Vec<i64> = action_space.getattr("shape")?.extract()?;
            action_space[0]
        };
        let observation_space = env.getattr("observation_space")?;
        let observation_space = observation_space.getattr("shape")?.extract()?;

        Ok(PyGymEnv {
            render: false,
            env: env.into(),
            action_space,
            observation_space,
            // TODO: consider remove RefCell, raw value instead
            count_steps: RefCell::new(0),
            max_steps: None,
            obs_filter,
            act_filter,
            phantom: PhantomData,
        })
    }

    pub fn set_render(&mut self, render: bool) {
        self.render = render;
    }

    pub fn max_steps(mut self, v: Option<usize>) -> Self {
        self.max_steps = v;
        self
    }    
}

impl<O, A, OF, AF> Env for PyGymEnv<O, A, OF, AF> where
    O: Obs,
    A: Act + Debug,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, returning the observation tensor.
    /// In this environment, the length of `is_done` is assumed to be 1.
    fn reset(&mut self, is_done: Option<&Vec<f32>>) -> Result<O, Box<dyn Error>>  {
        trace!("PyGymEnv::reset()");

        // Reset action filter, effective for stateful filter
        self.act_filter.reset(&is_done);

        match is_done {
            None => {
                pyo3::Python::with_gil(|py| {
                    let obs = self.env.call_method0(py, "reset")?;
                    Ok(self.obs_filter.reset(obs))
                })
            },
            Some(v) => {
                if v[0] == 0.0 as f32 {
                    Ok(O::zero(1))
                }
                else {
                    self.count_steps.replace(0);
                    pyo3::Python::with_gil(|py| {
                        let obs = self.env.call_method0(py, "reset")?;
                        Ok(self.obs_filter.reset(obs))
                    })
                }
            }
        }
    }

    fn step(&mut self, a: &A) -> (Step<Self>, Record) {
        trace!("PyGymEnv::step()");

        pyo3::Python::with_gil(|py| {
            if self.render {
                let _ = self.env.call_method0(py, "render");
            }

            let (a_py, record_a) = self.act_filter.filt(a.clone());
            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();
            let obs = step.get_item(0).to_owned();
            let (obs, record_o) = self.obs_filter.filt(obs.to_object(py));
            let reward: Vec<f32> = vec![step.get_item(1).extract().unwrap()];
            let mut is_done: Vec<f32> = vec![
                if step.get_item(2).extract().unwrap() {1.0} else {0.0}
            ];

            let c = *self.count_steps.borrow();
            self.count_steps.replace(c + 1);
            if let Some(max_steps) = self.max_steps {
                if *self.count_steps.borrow() >= max_steps {
                    is_done[0] = 1.0;
                }
            };

            (
                Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo{}),
                record_o.merge(record_a)
            )
        })
    }
}
