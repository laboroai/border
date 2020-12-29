use pyo3::{Python};
use lrr::core::{Trainer};
use lrr::py_gym_env::{PyGymEnv, PyGymDiscreteAct, PyNDArrayObs,
                      PyNDArrayObsAdapter, PyGymDiscreteActAdapter};
use lrr::agents::{DQN, dqn::QNetwork};

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
    let dqn: DQN<PyGymEnv<PyGymDiscreteAct>, _, _, _> = DQN::new(
        qnet,
        0,
        0,
        PyNDArrayObsAdapter::new(),
        PyGymDiscreteActAdapter::new()
    );
    let mut trainer = Trainer::new(env, dqn);

    // trainer.train();
}
