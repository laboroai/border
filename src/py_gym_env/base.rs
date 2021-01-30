#![allow(clippy::float_cmp)]
use std::borrow::Borrow;
use std::{fmt::Debug, error::Error};
use std::marker::PhantomData;
use std::cell::RefCell;
use log::{trace};
use pyo3::{PyObject, PyResult, Python, ToPyObject};
use pyo3::types::{PyTuple, IntoPyDict, PyList};
use crate::core::{Obs, Act, Info, Step, Env};

pub struct PyGymInfo {}

impl Info for PyGymInfo {}


/// Convert PyObject to Env::Obs.
///
/// The methods in this trait are called inside PyGymEnv.
pub trait ObsFilter<O: Obs> {
    fn filt(&self, obs: PyObject) -> O;

    fn reset(&self, obs: PyObject) -> O {
        self.filt(obs)
    }

    /// Stack filtered observations into an observation.
    ///
    /// This method is used in vectorized environments.
    fn stack(filtered: Vec<O>) -> O;
}

/// Adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning).
/// It represents non-vectorized environment (`n_procs`=1).
#[derive(Debug, Clone)]
pub struct PyGymEnv<O, A, F> where
    O: Obs,
    F: ObsFilter<O>
{
    render: bool,
    env: PyObject,
    action_space: i64,
    observation_space: Vec<usize>,
    continuous_action: bool,
    count_steps: RefCell<usize>,
    max_steps: Option<usize>,
    obs_filter: F,
    phantom: PhantomData<(O, A, F)>,
}

impl<O, A, F> PyGymEnv<O, A, F> where
    O: Obs,
    A: Act + Into<PyObject>,
    F: ObsFilter<O>,
{
        pub fn new(name: &str, obs_filter: F, continuous_action: bool) -> PyResult<Self> {
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
            continuous_action,
            count_steps: RefCell::new(0),
            max_steps: None,
            obs_filter,
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

fn pylist_to_act(py: &Python, a: PyObject) -> PyObject {
    let a_py_type = a.as_ref(*py).borrow().get_type().name().unwrap();
    if a_py_type == "list" {
        let l: &PyList = a.extract(*py).unwrap();
        l.get_item(0).into()
    }
    else {
        a
    }
}

impl<O, A, F> Env for PyGymEnv<O, A, F> where
    O: Obs,
    A: Act + Into<PyObject> + Debug,
    F: ObsFilter<O>,
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, returning the observation tensor.
    /// In this environment, the length of `is_done` is assumed to be 1.
    fn reset(&self, is_done: Option<&Vec<f32>>) -> Result<O, Box<dyn Error>>  {
        trace!("PyGymEnv::reset()");
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

    fn step(&self, a: &A) -> Step<Self> {
        trace!("PyGymEnv.step()");        
        trace!("a     : {:?}", &a);

        pyo3::Python::with_gil(|py| {
            if self.render {
                let _ = self.env.call_method0(py, "render");
            }

            // Process action for continuous or discrete
            let a_py = {
                let a_py = a.clone().into();
                if !self.continuous_action { pylist_to_act(&py, a_py) }
                else { a_py }
            };

            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();

            let obs = step.get_item(0).to_owned();
            let obs = self.obs_filter.filt(obs.to_object(py));
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

            Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo{})
        })
    }
}
