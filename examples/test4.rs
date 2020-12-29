use pyo3::{Python};
use lrr::core::{Policy, Sampler};
use lrr::py_gym_env::{PyGymEnv, PyGymDiscreteAct, PyNDArrayObs};
use lrr::agents::{DQN, dqn::QNetwork};
// struct MyDiscreteRandomPolicy {}

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
    let qnet = QNetwork::new(4, 2, 0.01);
    let pi = DQN::new(qnet, 0, 0);
    let sampler = Sampler::new(env, pi);

    sampler.sample(100);
}
