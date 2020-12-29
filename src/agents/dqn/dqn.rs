use tch::{Tensor, nn::Module, Kind::Float};
use crate::core::{Policy, Agent, Step};
use crate::py_gym_env::{PyGymEnv, PyNDArrayObs, PyGymDiscreteAct, PyGymInfo};

pub struct DQN<M: Module + Clone> {
    n_samples_per_opt: usize,
    n_updates_per_opt: usize,
    qnet: M,
    qnet_tgt: M,
    train: bool,
}

impl<M: Module + Clone> DQN<M> {
    pub fn new(qnet: M, n_samples_per_opt: usize, n_updates_per_opt: usize) -> Self {
        let qnet_tgt = qnet.clone();
        DQN {
            n_samples_per_opt,
            n_updates_per_opt,
            qnet,
            qnet_tgt,
            train: false
        }
    }
}

impl<M: Module + Clone> Policy<PyGymEnv<PyGymDiscreteAct>> for DQN<M> {
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn sample(&self, obs: &PyNDArrayObs) -> PyGymDiscreteAct {
        let obs = obs.0.view().to_slice().unwrap();
        let obs: Tensor = Tensor::of_slice(obs);
        let a = obs.apply(&self.qnet);
        let a: i32 = if self.train {
            a.softmax(-1, Float)
            .multinomial(1, false)
            .into()
        } else {
            a.argmax(-1, true).into()
        };
        PyGymDiscreteAct::new(a as u32)
    }
}

impl<M: Module + Clone> Agent<PyGymEnv<PyGymDiscreteAct>> for DQN<M> {
    fn observe(&self, _step: Step<PyNDArrayObs, PyGymInfo>) -> bool {
        true
    }
}
