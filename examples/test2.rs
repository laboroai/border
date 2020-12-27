use pyo3::{Python};
use lrr::core::{Policy, Sampler};
use lrr::py_gym_env::{PyGymEnv, PyGymDiscreteAct, PyNDArrayObs};

struct MyDiscreteRandomPolicy {}

impl Policy<PyGymEnv<PyGymDiscreteAct>> for MyDiscreteRandomPolicy {
    fn sample(&self, _: &PyNDArrayObs) -> PyGymDiscreteAct {
        PyGymDiscreteAct::new(fastrand::u32(..=1))
    }
}

fn main() {
    let mut env = match PyGymEnv::<PyGymDiscreteAct>::new("CartPole-v0") {
        Ok(env) => env,
        Err(e) => {
            let gil = Python::acquire_gil();
            let py = gil.python();
            println!("{:?}", e.ptype(py));
            println!("{:?}", e.pvalue(py));
            panic!();
        }
    };
    env.set_render(false);
    let pi = MyDiscreteRandomPolicy {};
    let sampler = Sampler::new(env, pi);

    sampler.sample(100);
}
