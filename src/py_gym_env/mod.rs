// use tch::Tensor;
use ndarray::{ArrayD, IxDyn}; //array};
use cpython::{NoArgs, ObjectProtocol, PyObject, PyResult, Python, ToPyObject};
use crate::core::{Info, Obs};

pub struct PyInfo {}

impl Info for PyInfo {}

#[derive(Clone)]
pub struct PyNDArrayObs (ArrayD<f32>);

impl Obs for PyNDArrayObs {
    fn new() -> Self {
        PyNDArrayObs(ArrayD::<f32>::zeros(IxDyn(&[1])))
    }
}

pub struct PyNDArrayAct (ArrayD<f32>);

/// Adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning)
pub struct PyGymEnv {
    env: PyObject,
    action_space: i64,
    observation_space: Vec<i64>,
}

impl PyGymEnv {
    fn new(name: &str) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let gym = py.import("gym")?;
        let env = gym.call(py, "make", (name,), None)?;
        let _ = env.call_method(py, "seed", (42,), None)?;
        let action_space = env.getattr(py, "action_space")?;
        let action_space = if let Ok(val) = action_space.getattr(py, "n") {
            val.extract(py)?
        } else {
            let action_space: Vec<i64> = action_space.getattr(py, "shape")?.extract(py)?;
            action_space[0]
        };
        let observation_space = env.getattr(py, "observation_space")?;
        let observation_space = observation_space.getattr(py, "shape")?.extract(py)?;
        Ok(PyGymEnv {
            env,
            action_space,
            observation_space,
        })
    }

    /// Resets the environment, returning the observation tensor.
    pub fn reset(&self) -> () { //PyResult<Tensor> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obs = self.env.call_method(py, "reset", NoArgs, None)?;
        // Ok(Tensor::of_slice(&obs.extract::<Vec<f32>>(py)?));
    }    
}
