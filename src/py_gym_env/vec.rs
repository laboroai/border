/// Vectorized version of the gym environment.
/// Adapted from tch-rs RL example.
use std::{fmt::Debug, error::Error};
use std::marker::PhantomData;
use log::{trace};
use pyo3::{PyObject, PyResult, ToPyObject};
use pyo3::types::{PyTuple, IntoPyDict};
use crate::core::{Obs, Act, Step, Env};
use crate::py_gym_env::PyGymInfo;

#[derive(Debug, Clone)]
pub struct PyVecGymEnv<O, A> {
    env: PyObject,
    // action_space: i64,
    // observation_space: Vec<usize>,
    phantom: PhantomData<(O, A)>,
}

impl<O, A> PyVecGymEnv<O, A> where 
    O: Obs + From<PyObject>,
    A: Act + Into<PyObject>,
{
    pub fn new(name: &str, img_dir: Option<&str>, nprocesses: i64) -> PyResult<Self> {
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
            let env = gym.call("make", (name, img_dir, nprocesses), None)?;

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
                // action_space,
                // observation_space,
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

impl<O, A> Env for PyVecGymEnv<O, A> where
    O: Obs + From<PyObject>,
    A: Act + Into<PyObject>,
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, returning the observation tensor.
    fn reset(&self, is_done: Option<&Vec<f32>>) -> Result<O, Box<dyn Error>>  {
        unimplemented!();
        // pyo3::Python::with_gil(|py| {
        //     // let gil = Python::acquire_gil();
        //     // let py = gil.python();
        //     let obs = self.env.call_method0(py, "reset")?;
        //     Ok(obs.into())
        // })
    }    
    
    fn step(&self, a: &A) -> Step<Self> {
        trace!("{:?}", &a);
        pyo3::Python::with_gil(|py| {
            let a_py = a.clone().into();
            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();
            let obs = step.get_item(0).to_owned();
            let obs = obs.to_object(py).into();
            // let reward: f32 = step.get_item(1).extract().unwrap();
            // let is_done: bool = step.get_item(2).extract().unwrap();

            let reward = Vec::<f32>::new();
            let is_done = Vec::<f32>::new();
            unimplemented!();

            Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo{})
        })
    }
}
