use std::error::Error;
use ndarray::ArrayD;
use pyo3::{PyObject, IntoPy};
use numpy::PyArrayDyn;
use lrr::core::{Obs, Act, Policy, util};
use lrr::py_gym_env::PyGymEnv;

#[derive(Clone, Debug)]
pub struct CartPoleObs (pub ArrayD<f32>);

impl Obs for CartPoleObs {}

impl From<PyObject> for CartPoleObs {
    fn from(obs: PyObject) -> Self {
        pyo3::Python::with_gil(|py| {
            let obs: &PyArrayDyn<f64> = obs.extract(py).unwrap();
            let obs = obs.to_owned_array();
            let obs = obs.mapv(|elem| elem as f32);
            Self {
                0: obs
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct CartPoleAct (u32);

impl Act for CartPoleAct {}

impl Into<PyObject> for CartPoleAct {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            self.0.into_py(py)
        })
    }
}

impl CartPoleAct {
    pub fn new(v: u32) -> Self {
        CartPoleAct { 0: v }
    }
}

type CartPoleEnv = PyGymEnv::<CartPoleObs, CartPoleAct>;

struct RandomPolicy {}

impl Policy<CartPoleEnv> for RandomPolicy {
    fn sample(&self, _: &CartPoleObs) -> CartPoleAct {
        CartPoleAct::new(fastrand::u32(..=1))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    tch::manual_seed(42);

    let env = CartPoleEnv::new("CartPole-v0")?;
    let policy = RandomPolicy{};
    util::eval(&env, &policy, 5, None);

    Ok(())
}
