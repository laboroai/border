use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{Kind::Float, Tensor};

use crate::{
    core::{
        Policy, Agent, Step, Env,
        record::{Record, RecordValue}
    },
    agent::{
        OptInterval, OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer, TchBatch,
            model::Model1
        }
    }
};

pub struct PGDiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor>, // + Clone
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    opt_interval_counter: OptIntervalCounter,
    n_updates_per_opt: usize,
    batch_size: usize,
    model: M,
    train: bool,
    prev_obs: RefCell<Option<E::Obs>>,
    replay_buffer: ReplayBuffer<E, O, A>,
    discount_factor: f64,
    phandom: PhantomData<E>,
}

impl<E, M, O, A> PGDiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor>, // + Clone
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    pub fn new(model: M, replay_buffer: ReplayBuffer<E, O, A>) -> Self {
        PGDiscrete {
            opt_interval_counter: OptInterval::Episodes(1).counter(),
            n_updates_per_opt: 1,
            batch_size: 1,
            model,
            train: false,
            replay_buffer,
            discount_factor: 0.99,
            prev_obs: RefCell::new(None),
            phandom: PhantomData,
        }
    }

    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval_counter = v.counter();
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

    fn update_model(&mut self, batch: TchBatch<E, O, A>) -> f32 {
        trace!("PGDiscrete::update_qnet()");

        // adapted from ppo.rs in tch-rs RL example
        trace!("batch.obs.shape      = {:?}", &batch.obs.size());
        trace!("batch.next_obs.shape = {:?}", &batch.next_obs.size());
        trace!("batch.actions.shape  = {:?}", &batch.actions.size());
        trace!("batch.rewards.shape  = {:?}", &batch.rewards.size());
        trace!("batch.returns.shape  = {:?}", &batch.returns.as_ref().unwrap().size());

        let loss = {
            let actor = self.model.forward(&batch.obs);
            let log_probs = actor.log_softmax(-1, tch::Kind::Float);
            let action_log_probs = {
                let index = batch.actions; //.to_device(device);
                log_probs.gather(-1, &index, false).squeeze1(-1)
            };
            (-batch.returns.unwrap().detach() * action_log_probs).mean(tch::Kind::Float)
        };

        self.model.backward_step(&loss);

        f32::from(loss)
    }
}

impl <E, M, O, A> Policy<E> for PGDiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor>, // + Clone,
    E::Obs :Into<M::Input> + Clone,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let a = self.model.forward(&obs);
        let a = if self.train {
            a.softmax(-1, Float)
            .multinomial(1, true)
        } else {
            a.argmax(-1, true)
        };
        a.into()
    }
}

impl <E, M, O, A> Agent<E> for PGDiscrete<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor>, // + Clone
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

    /// Update model parameters.
    ///
    /// When the return value is `Some(Record)`, it includes:
    /// * `loss`: Loss for poligy gradient
    fn observe(&mut self, step: Step<E>) -> Option<Record> {
        trace!("PGDiscrete.observe()");

        // Check if doing optimization
        let do_optimize = self.opt_interval_counter.do_optimize(&step.is_done);
            // && self.replay_buffer.len() + 1 >= self.min_transitions_warmup;

        // Push transition to the replay buffer
        self.push_transition(step);
        trace!("Push transition");

        // Do optimization
        if do_optimize {
            let mut loss = 0f64;

            // Store returns in the replay buffer
            // let (estimated_return, _)
            //     = self.model.forward(&self.prev_obs.borrow().to_owned().unwrap().into());
            // self.replay_buffer.update_returns(estimated_return.detach(), self.discount_factor);
            let estimated_return = Tensor::of_slice(&[0f32]);
            self.replay_buffer.update_returns(estimated_return, self.discount_factor);

            // Update model parameters
            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                loss += self.update_model(batch) as f64;
            };

            // Clear replay buffer
            self.replay_buffer.clear();

            loss /= self.n_updates_per_opt as f64;

            Some(Record::from_slice(&[
                ("loss", RecordValue::Scalar(loss)),
            ]))
        }
        else {
            None
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        self.model.save(&path.as_ref().join("model.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.model.load(&path.as_ref().join("model.pt").as_path())?;
        Ok(())
    }
}
