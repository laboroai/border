use std::cell::RefCell;
use std::marker::PhantomData;
use tch::{Tensor, nn::Module, Kind::Float};
use crate::{agents::ReplayBuffer, core::{Policy, Agent, Step, Env}};
use crate::agents::{ModuleActAdapter, ModuleObsAdapter};

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
    prev_obs: RefCell<Option<Tensor>>,
    replay_buffer: ReplayBuffer<E, I, O>
}

impl<E, M, I, O> DQN<E, M, I, O> where 
    E: Env,
    M: Module + Clone,
    I: ModuleObsAdapter<E::Obs>,
    O: ModuleActAdapter<E::Act> {
    pub fn new(qnet: M, replay_buffer: ReplayBuffer<E, I, O>, n_samples_per_opt: usize,
               n_updates_per_opt: usize, from_obs: I, into_act: O) -> Self {
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
            prev_obs: RefCell::new(None),
            replay_buffer,
        }
    }

    fn push_transition(&mut self, step: Step<E::Obs, E::Act, E::Info>) {
        let next_obs = self.from_obs.convert(&step.obs);
        let obs = self.prev_obs.replace(None).unwrap();
        self.replay_buffer.push(
            &obs,
            &self.into_act.back(&step.act),
            // TODO: check if reward have a dimension, not scalar
            &step.reward.into(),
            &next_obs
        );
        let _ = self.prev_obs.replace(Some(next_obs));
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

impl<E, M, I, O> Agent<E> for DQN<E, M, I, O> where
    E: Env,
    M: Module + Clone,
    I: ModuleObsAdapter<E::Obs>,
    O: ModuleActAdapter<E::Act> {

    fn push_obs(&self, obs: &E::Obs) {
        self.prev_obs.replace(Some(self.from_obs.convert(obs)));
    }

    fn observe(&mut self, step: Step<E::Obs, E::Act, E::Info>) -> bool {
        // Push transition to the replay buffer
        self.push_transition(step);
        true
    }
}
