use std::cell::RefCell;
use std::marker::PhantomData;
use tch::{no_grad, Kind::Float, Tensor};
use crate::{agents::{ReplayBuffer, Model}, core::{Policy, Agent, Step, Env}};
use crate::agents::{TchActAdapter, TchObsAdapter};
use crate::agents::tch::util::track;

pub struct DQN<E, M, I, O> where
    E: Env,
    M: Model + Clone,
    I: TchObsAdapter<E::Obs>,
    O: TchActAdapter<E::Act> {
    n_samples_per_opt: usize,
    n_updates_per_opt: usize,
    n_opts_per_soft_update: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    qnet: M,
    qnet_tgt: M,
    from_obs: I,
    into_act: O,
    train: bool,
    phantom: PhantomData<E>,
    prev_obs: RefCell<Option<Tensor>>,
    replay_buffer: ReplayBuffer<E, I, O>,
    count_samples_per_opt: usize,
    count_opts_per_soft_update: usize,
    discount_factor: f64,
    tau: f64,
}

impl<E, M, I, O> DQN<E, M, I, O> where 
    E: Env,
    M: Model + Clone,
    I: TchObsAdapter<E::Obs>,
    O: TchActAdapter<E::Act> {

    #[allow(clippy::too_many_arguments)]
    pub fn new(qnet: M, replay_buffer: ReplayBuffer<E, I, O>, from_obs: I, into_act: O)
            -> Self {
        let qnet_tgt = qnet.clone();
        DQN {
            qnet,
            qnet_tgt,
            replay_buffer,
            from_obs,
            into_act,
            n_samples_per_opt: 1,
            n_updates_per_opt: 1,
            n_opts_per_soft_update: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            count_samples_per_opt: 0,
            count_opts_per_soft_update: 0,
            train: false,
            prev_obs: RefCell::new(None),
            phantom: PhantomData,
        }
    }

    pub fn n_samples_per_opt(mut self, v: usize) -> Self {
        self.n_samples_per_opt = v;
        self
    }

    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    pub fn n_opts_per_soft_update(mut self, v: usize) -> Self {
        self.n_opts_per_soft_update = v;
        self
    }

    pub fn min_transitions_warmup(mut self, v: usize) -> Self {
        self.min_transitions_warmup = v;
        self
    }

    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn discount_factor(mut self, v: f64) -> Self {
        self.discount_factor = v;
        self
    }

    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
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

    fn update_qnet(&mut self, batch: (Tensor, Tensor, Tensor, Tensor)) {
        let (obs, a, r, next_obs) = batch;
        let loss = {
            let pred = {
                let a = a;
                let x = self.qnet.forward(&obs);
                x.gather(-1, &a, false)
            };
            let tgt = no_grad(|| {
                let x = self.qnet_tgt.forward(&next_obs);
                let y = x.argmax(-1, false).unsqueeze(-1);
                let x = x.gather(-1, &y, false);
                r + self.discount_factor * x
            });
            pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0)
        };
        self.qnet.backward_step(&loss);
    }

    fn soft_update_qnet_tgt(&mut self) {
        track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
    }
}

impl<E, M, I, O> Policy<E> for DQN<E, M, I, O> where 
    E: Env,
    M: Model + Clone,
    I: TchObsAdapter<E::Obs>,
    O: TchActAdapter<E::Act> {
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn sample(&self, obs: &E::Obs) -> E::Act {
        let obs = self.from_obs.convert(obs);
        let a = self.qnet.forward(&obs);
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
    M: Model + Clone,
    I: TchObsAdapter<E::Obs>,
    O: TchActAdapter<E::Act> {

    fn push_obs(&self, obs: &E::Obs) {
        self.prev_obs.replace(Some(self.from_obs.convert(obs)));
    }

    fn observe(&mut self, step: Step<E::Obs, E::Act, E::Info>) -> bool {
        // Push transition to the replay buffer
        self.push_transition(step);

        // Do optimization 1 step
        self.count_samples_per_opt += 1;
        if self.count_samples_per_opt == self.n_samples_per_opt {
            self.count_samples_per_opt = 0;

            if self.replay_buffer.len() >= self.min_transitions_warmup {
                for _ in 0..self.n_updates_per_opt {
                    let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                    self.update_qnet(batch);
                };

                self.count_opts_per_soft_update += 1;
                if self.count_opts_per_soft_update == self.n_opts_per_soft_update {
                    self.count_opts_per_soft_update = 0;
                    self.soft_update_qnet_tgt();
                }
                return true;
            }
        }
        false
    }
}
