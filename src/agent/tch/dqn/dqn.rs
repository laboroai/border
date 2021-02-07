use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{no_grad, Kind::Float, Tensor};
use crate::core::{Policy, Agent, Step, Env};
use crate::agents::{OptInterval, OptIntervalCounter};
use crate::agents::tch::{ReplayBuffer, TchBuffer, TchBatch};
use crate::agents::tch::model::Model1;
use crate::agents::tch::util::track;

pub struct DQN<E, M, O, A> where
    E: Env,
    M: Model1 + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    opt_interval_counter: OptIntervalCounter,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    qnet: M,
    qnet_tgt: M,
    train: bool,
    phantom: PhantomData<E>,
    prev_obs: RefCell<Option<E::Obs>>,
    replay_buffer: ReplayBuffer<E, O, A>,
    discount_factor: f64,
    tau: f64,
}

impl<E, M, O, A> DQN<E, M, O, A> where 
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(qnet: M, replay_buffer: ReplayBuffer<E, O, A>)
            -> Self {
        let qnet_tgt = qnet.clone();
        DQN {
            qnet,
            qnet_tgt,
            replay_buffer,
            opt_interval_counter: OptInterval::Steps(1).counter(),
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            train: false,
            prev_obs: RefCell::new(None),
            phantom: PhantomData,
        }
    }

    // pub fn opt_interval(mut self, v: OptInterval) -> Self {
    //     self.opt_interval = v;
    //     self
    // }

    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval_counter = v.counter();
        self
    }

    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
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

    fn push_transition(&mut self, step: Step<E>) {
        trace!("DQN::push_transition()");

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

    fn update_critic(&mut self, batch: TchBatch<E, O, A>) {
        trace!("DQN::update_critic()");

        let obs = batch.obs;
        let a = batch.actions;
        let r = batch.rewards;
        let next_obs = batch.next_obs;
        let not_done = batch.not_dones;
        trace!("obs.shape      = {:?}", obs.size());
        trace!("next_obs.shape = {:?}", next_obs.size());
        trace!("a.shape        = {:?}", a.size());
        trace!("r.shape        = {:?}", r.size());
        trace!("not_done.shape = {:?}", not_done.size());

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
                r + not_done * self.discount_factor * x
            });
            pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0)
        };
        self.qnet.backward_step(&loss);
    }

    fn soft_update(&mut self) {
        trace!("DQN::soft_update()");
        track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
    }
}

impl<E, M, O, A> Policy<E> for DQN<E, M, O, A> where 
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let a = self.qnet.forward(&obs);
        let a = if self.train {
            a.softmax(-1, Float)
            .multinomial(1, true)
        } else {
            a.argmax(-1, true)
        };
        a.into()
    }
}

impl<E, M, O, A> Agent<E> for DQN<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
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
        trace!("DQN.observe()");

        // Check if doing optimization
        let do_optimize = self.opt_interval_counter.do_optimize(&step.is_done)
            && self.replay_buffer.len() + 1 >= self.min_transitions_warmup;

        // Push transition to the replay buffer
        self.push_transition(step);
        trace!("Push transition");

        // Do optimization
        if do_optimize {
            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                trace!("Sample random batch");

                self.update_critic(batch);
                self.soft_update();
                trace!("Update model");
            };
            true
        }
        else {
            false
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        fs::create_dir(&path)?;
        self.qnet.save(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt.save(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.qnet.load(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt.load(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }
}
