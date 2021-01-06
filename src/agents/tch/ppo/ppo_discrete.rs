use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{Kind::Float, Tensor};
use crate::core::{Policy, Agent, Step, Env};
use crate::agents::{ReplayBuffer, TchBufferableActInfo, TchBufferableObsInfo};
use crate::agents::tch::model::Model;
use crate::agents::tch::Batch2;

pub struct PPODiscrete<E, M> where
    E: Env,
    M: Model<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone
    E::Obs :TchBufferableObsInfo + Into<M::Input>,
    E::Act :TchBufferableActInfo + Into<Tensor> + From<Tensor> {

    n_samples_per_opt: usize, // NOTE: must be equal to replay buffer size
    n_updates_per_opt: usize,
    batch_size: usize,
    model: M,
    train: bool,
    prev_obs: RefCell<Option<Tensor>>,
    replay_buffer: ReplayBuffer<E>,
    count_samples_per_opt: usize,
    discount_factor: f64,
    phandom: PhantomData<E>,
}

impl<E, M> PPODiscrete<E, M> where
    E: Env,
    M: Model<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone
    E::Obs :TchBufferableObsInfo + Into<M::Input>,
    E::Act :TchBufferableActInfo + Into<Tensor> + From<Tensor> {

    pub fn new(model: M, n_samples_per_opt: usize) -> Self {
        let replay_buffer = ReplayBuffer::new(n_samples_per_opt);
        PPODiscrete {
            n_samples_per_opt,
            n_updates_per_opt: 1,
            batch_size: 1,
            model,
            train: false,
            replay_buffer,
            count_samples_per_opt: 0,
            discount_factor: 0.99,
            prev_obs: RefCell::new(None),
            phandom: PhantomData,
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

    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn discount_factor(mut self, v: f64) -> Self {
        self.discount_factor = v;
        self
    }

    fn push_transition(&mut self, step: Step<E>) {
        let next_obs = step.obs.into();
        let obs = self.prev_obs.replace(None).unwrap();
        let not_done = (if step.is_done { 0.0 } else { 1.0 }).into();
        self.replay_buffer.push(
            &obs,
            &step.act.clone().into(),
            &step.reward.into(),
            &next_obs,
            &not_done,
        );
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn update_model(&mut self, batch: Batch2) {
        // adapted from ppo.rs in tch-rs RL example
        let (critic, actor) = self.model.forward(&batch.obs);
        let log_probs = actor.log_softmax(-1, tch::Kind::Float);
        let probs = actor.softmax(-1, tch::Kind::Float);
        let action_log_probs = {
            let index = batch.actions; //.to_device(device);
            log_probs.gather(-1, &index, false).squeeze1(-1)
        };
        let dist_entropy = (-log_probs * probs)
            .sum1(&[-1], false, tch::Kind::Float)
            .mean(tch::Kind::Float);
        let advantages = batch.returns - critic;
        let value_loss = (&advantages * &advantages).mean(tch::Kind::Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(tch::Kind::Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.model.backward_step(&loss);
    }
}

impl <E, M> Policy<E> for PPODiscrete<E, M> where
    E: Env,
    M: Model<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone,
    E::Obs :TchBufferableObsInfo + Into<M::Input>,
    E::Act :TchBufferableActInfo + Into<Tensor> + From<Tensor> {

    fn sample(&self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let (_, a) = self.model.forward(&obs);
        let a = if self.train {
            a.softmax(-1, Float)
            .multinomial(1, true)
        } else {
            a.argmax(-1, true)
        };
        a.into()
    }
}

impl <E, M> Agent<E> for PPODiscrete<E, M> where
    E: Env,
    M: Model<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone
    E::Obs :TchBufferableObsInfo + Into<M::Input>,
    E::Act :TchBufferableActInfo + Into<Tensor> + From<Tensor> {

    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn push_obs(&self, obs: &E::Obs) {
        self.prev_obs.replace(Some(obs.clone().into()));
    }

    fn observe(&mut self, step: Step<E>) -> bool {
        // Push transition to the replay buffer
        self.push_transition(step);

        // Do optimization 1 step
        self.count_samples_per_opt += 1;
        if self.count_samples_per_opt == self.n_samples_per_opt {
            self.count_samples_per_opt = 0;

            // Store returns in the replay buffer
            let (estimated_return, _)
                = self.model.forward(&self.prev_obs.borrow().as_ref().unwrap());
            self.replay_buffer.update_returns(estimated_return, self.discount_factor);

            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch2(self.batch_size).unwrap();
                self.update_model(batch);
            };
            return true;
        }
        false
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        fs::create_dir(&path)?;
        self.model.save(&path.as_ref().join("model.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.model.load(&path.as_ref().join("model.pt").as_path())?;
        Ok(())
    }
}
