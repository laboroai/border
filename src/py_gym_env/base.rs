use std::fmt::Debug;
use std::marker::PhantomData;
use ndarray::{Array, ArrayD, IxDyn};
use pyo3::{IntoPy, PyErr, PyObject, PyResult, Python};
use pyo3::types::{PyTuple};
use numpy::{PyArrayDyn};
use crate::core::{Info, Obs, Step, Env};

pub struct PyGymInfo {}

impl Info for PyGymInfo {}

#[derive(Clone)]
pub struct PyNDArrayObs (pub ArrayD<f32>);

impl Obs for PyNDArrayObs {
    fn new() -> Self {
        PyNDArrayObs(ArrayD::<f32>::zeros(IxDyn(&[1])))
    }
}

pub trait PyGymEnvAct {
    fn to_pyobj(&self, py: Python) -> PyObject;
}

#[derive(Debug)]
pub struct PyGymDiscreteAct (u32);

impl PyGymDiscreteAct {
    pub fn new(v: u32) -> Self {
        PyGymDiscreteAct { 0: v }
    }
}

impl PyGymEnvAct for PyGymDiscreteAct {
    fn to_pyobj(&self, py: Python) -> PyObject {
        self.0.into_py(py)
    }
}

/// Adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning)
pub struct PyGymEnv<A> {
    render: bool,
    env: PyObject,
    action_space: i64,
    observation_space: Vec<usize>,
    action_type: PhantomData<A>,
}

impl<A: PyGymEnvAct + Debug> PyGymEnv<A> {
    pub fn new(name: &str) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
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
            action_type: PhantomData,
        })
    }

    pub fn set_render(&mut self, render: bool) {
        self.render = render;
    }
}

impl<A: PyGymEnvAct + Debug> Env for PyGymEnv<A> {
    type Obs = PyNDArrayObs;
    type Act = A;
    type Info = PyGymInfo;
    type ERR = PyErr;

    /// Resets the environment, returning the observation tensor.
    fn reset(&self) -> PyResult<PyNDArrayObs>  {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obs = self.env.call_method0(py, "reset")?;
        Ok(PyNDArrayObs(
            Array::from_shape_vec(
                IxDyn(&self.observation_space),
                obs.extract::<Vec<f32>>(py)?.clone()
            ).unwrap()
        ))
    }

    fn step(&self, a: &A) -> Step<PyNDArrayObs, PyGymInfo> {
            println!("{:?}", &a);
        pyo3::Python::with_gil(|py| {
            if self.render {
                let _ = self.env.call_method0(py, "render");
            }
            let a = a.to_pyobj(py);
            let ret = self.env.call_method(py, "step", (a,), None).unwrap();

            let step: &PyTuple = ret.extract(py).unwrap();

            let obs1: &PyArrayDyn<f64> = step.get_item(0).extract().unwrap();
            let obs2 = obs1.readonly();
            let obs3 = obs2.as_array();
            let obs4 = obs3.mapv(|elem| elem as f32);

            let r: f32 = step.get_item(1).extract().unwrap();
            let is_done: bool = step.get_item(2).extract().unwrap();

            Step::new(PyNDArrayObs(obs4), r, is_done, PyGymInfo{})
        })
    }
}
