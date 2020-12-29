use std::marker::PhantomData;
use tch::{Tensor, nn::Module, Kind::Float};
use crate::core::{Obs, Policy, Agent, Step, Env};
use crate::py_gym_env::{PyGymEnv, PyNDArrayObs, PyGymDiscreteAct, PyGymInfo};

pub trait ModuleObsAdapter<T: Obs> {
    fn convert(&self, obs: &T) -> Tensor;
}

pub trait ModuleActAdapter<T> {
    fn convert(&self, act: &Tensor) -> T;
}

pub struct DQN<E, M, I, O> where
    E: Env,
    M: Module + Clone,
    I: ModuleObsAdapter<E::Obs>,
    O: ModuleActAdapter<E::Act> {
    n_samples_per_opt: usize,
    n_updates_per_opt: usize,
    qnet: M,
    qnet_tgt: M,
    from_obs: I,
    into_act: O,
    train: bool,
    phantom: PhantomData<E>,
}

impl<E, M, I, O> DQN<E, M, I, O> where 
    E: Env,
    M: Module + Clone,
    I: ModuleObsAdapter<E::Obs>,
    O: ModuleActAdapter<E::Act> {
    pub fn new(qnet: M, n_samples_per_opt: usize, n_updates_per_opt: usize,
               from_obs: I, into_act: O) -> Self {
        let qnet_tgt = qnet.clone();
        DQN {
            n_samples_per_opt,
            n_updates_per_opt,
            qnet,
            qnet_tgt,
            from_obs,
            into_act,
            train: false,
            phantom: PhantomData,
        }
    }
}

impl<E, M, I, O> Policy<E> for DQN<E, M, I, O> where 
    E: Env,
    M: Module + Clone,
    I: ModuleObsAdapter<E::Obs>,
    O: ModuleActAdapter<E::Act> {
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    // fn sample(&self, obs: &PyNDArrayObs) -> PyGymDiscreteAct {
    //     let obs = obs.0.view().to_slice().unwrap();
    //     let obs: Tensor = Tensor::of_slice(obs);
    //     let a = obs.apply(&self.qnet);
    //     let a: i32 = if self.train {
    //         a.softmax(-1, Float)
    //         .multinomial(1, false)
    //         .into()
    //     } else {
    //         a.argmax(-1, true).into()
    //     };
    //     PyGymDiscreteAct::new(a as u32)
    // }
    fn sample(&self, obs: &E::Obs) -> E::Act {
        let obs = self.from_obs.convert(obs);
        let a = obs.apply(&self.qnet);
        let a = if self.train {
            a.softmax(-1, Float)
            .multinomial(1, false)
        } else {
            a.argmax(-1, true)
        };
        self.into_act.convert(&a)
    }
}

// impl<M> Agent<PyGymEnv<PyGymDiscreteAct>> for DQN<E, M, I, O> where
//     E: Env,
//     M: Module + Clone,
//     I: ModuleInputAdapter<E>,
//     O: ModuleOutputAdapter<E> {
//     fn observe(&self, _step: Step<PyNDArrayObs, PyGymInfo>) -> bool {
//         true
//     }
// }
