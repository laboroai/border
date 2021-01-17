use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{Kind::Float, Tensor};
use crate::core::{Policy, Agent, Step, Env};
use crate::agents::tch::{ReplayBuffer, TchBuffer, TchBatch};
use crate::agents::tch::model::Model1;

pub struct PPODiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    n_samples_per_opt: usize, // NOTE: must be equal to replay buffer size
    n_updates_per_opt: usize,
    batch_size: usize,
    model: M,
    train: bool,
    prev_obs: RefCell<Option<E::Obs>>,
    replay_buffer: ReplayBuffer<E, O, A>,
    count_samples_per_opt: usize,
    discount_factor: f64,
    phandom: PhantomData<E>,
}

impl<E, M, O, A> PPODiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    pub fn new(model: M, n_samples_per_opt: usize, n_procs: usize) -> Self {
        let capacity = n_samples_per_opt;
        let replay_buffer = ReplayBuffer::new(capacity, n_procs);
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
        let next_obs = step.obs;
        let obs = self.prev_obs.replace(None).unwrap();
        let reward = Tensor::of_slice(&step.reward[..]);
        let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);
        self.replay_buffer.push(
            &obs,
            &step.act,
            &reward,
            &next_obs,
            &not_done,
        );
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn update_model(&mut self, batch: TchBatch<E, O, A>) {
        trace!("Start ppo_discrete.update_qnet()");

        // adapted from ppo.rs in tch-rs RL example
        trace!("batch.obs.shape      = {:?}", &batch.obs.size());
        trace!("batch.next_obs.shape = {:?}", &batch.next_obs.size());
        trace!("batch.actions.shape  = {:?}", &batch.actions.size());
        trace!("batch.rewards.shape  = {:?}", &batch.rewards.size());
        trace!("batch.returns.shape  = {:?}", &batch.returns.as_ref().unwrap().size());

        let (critic, actor) = self.model.forward(&batch.obs);
        trace!("critic.shape        = {:?}", critic.size());
        trace!("actor.shape         = {:?}", actor.size());

        let log_probs = actor.log_softmax(-1, tch::Kind::Float);
        let probs = actor.softmax(-1, tch::Kind::Float);
        let action_log_probs = {
            let index = batch.actions; //.to_device(device);
            log_probs.gather(-1, &index, false).squeeze1(-1)
        };
        let dist_entropy = (-log_probs * probs)
            .sum1(&[-1], false, tch::Kind::Float)
            .mean(tch::Kind::Float);

        let advantages = batch.returns.unwrap() - critic;
        let value_loss = (&advantages * &advantages).mean(tch::Kind::Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(tch::Kind::Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;

        self.model.backward_step(&loss);
    }
}

impl <E, M, O, A> Policy<E> for PPODiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone,
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
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

impl <E, M, O, A> Agent<E> for PPODiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=(Tensor, Tensor)>, // + Clone
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
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
        self.prev_obs.replace(Some(obs.clone()));
    }

    fn observe(&mut self, step: Step<E>) -> bool {
        trace!("Start ppo_discrete.observe()");

        // Push transition to the replay buffer
        self.push_transition(step);

        // Do optimization 1 step
        self.count_samples_per_opt += 1;
        if self.count_samples_per_opt == self.n_samples_per_opt {
            self.count_samples_per_opt = 0;

            // Store returns in the replay buffer
            trace!("prev_obs.shape = {:?}", &self.prev_obs.borrow().to_owned().unwrap().into().size());
            let (estimated_return, _)
                = self.model.forward(&self.prev_obs.borrow().to_owned().unwrap().into());
            trace!("Call model.forward()");

            self.replay_buffer.update_returns(estimated_return, self.discount_factor);
            trace!("Update returns");

            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                trace!("Sample random batch");

                self.update_model(batch);
                trace!("Update model");
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
