//! DDPG agent.
use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{no_grad, Tensor};

use crate::{
    core::{
        Policy, Agent, Step, Env,
        record::{Record, RecordValue}
    },
    agent::{
        OptInterval, OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer, TchBatch,
            model::{Model1, Model2},
            util::track
        }
    }
};

type ActionValue = Tensor;

struct ActionNoise {
    mu: f64,
    theta: f64,
    sigma: f64,
    state: Tensor
}

impl ActionNoise {
    pub fn new() -> Self {
        let n_procs = 1;
        Self {
            mu: 0.0,
            theta: 0.15,
            sigma: 0.2,
            state: Tensor::ones(&[n_procs, 1], tch::kind::FLOAT_CPU),
        }
    }

    // pub fn update(&mut self) {
    //     // self.var = (self.var * 0.999).max(0.01);
    // }

    pub fn apply(&mut self, t: &Tensor) -> Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(&self.state.size(), tch::kind::FLOAT_CPU);
        self.state += dx;
        t + &self.state
        //self.var * Tensor::randn(t.size().as_slice(), tch::kind::FLOAT_CPU)
    }
}

/// adapted from ddpg.rs in tch-rs RL examples
pub struct DDPG<E, Q, P, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    critic: Q,
    critic_tgt: Q,
    actor: P,
    actor_tgt: P,
    action_noise: ActionNoise,
    replay_buffer: ReplayBuffer<E, O, A>,
    gamma: f64,
    tau: f64,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    opt_interval_counter: OptIntervalCounter,
    train: bool,
    prev_obs: RefCell<Option<E::Obs>>,
    phantom: PhantomData<E>
}

impl<E, Q, P, O, A> DDPG<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = A::SubBatch> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    /// Constructs DDPG agent.
    pub fn new(critic: Q, actor: P, replay_buffer: ReplayBuffer<E, O, A>) -> Self {
        let critic_tgt = critic.clone();
        let actor_tgt = actor.clone();
        DDPG {
            critic,
            critic_tgt,
            actor,
            actor_tgt,
            action_noise: ActionNoise::new(),
            replay_buffer,
            gamma: 0.99,
            tau: 0.005,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            opt_interval_counter: OptInterval::Steps(1).counter(),
            train: false,
            prev_obs: RefCell::new(None),
            phantom: PhantomData,
        }
    }

    /// Set optimization interval.
    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval_counter = v.counter();
        self
    }

    /// Set the number of updates per optimization step.
    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    /// Set the number of interaction steps for filling the replay buffer.
    pub fn min_transitions_warmup(mut self, v: usize) -> Self {
        self.min_transitions_warmup = v;
        self
    }

    /// Set the batch size.
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    /// Set the discount factor.
    pub fn discount_factor(mut self, v: f64) -> Self {
        self.gamma = v;
        self
    }

    /// Set the soft update coefficient.
    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    // Adapted from dqn.rs
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

    fn update_critic(&mut self, batch: &TchBatch<E, O, A>) -> f32 {
        trace!("DDPG.update_critic()");

        let loss = {
            let o = &batch.obs;
            let a = &batch.actions;
            let next_o = &batch.next_obs;
            let r = &batch.rewards;
            let not_done = &batch.not_dones;
            // trace!("obs.shape      = {:?}", o.size());
            // trace!("next_obs.shape = {:?}", next_o.size());
            // trace!("act.shape      = {:?}", a.size());
            trace!("reward.shape   = {:?}", r.size());
            trace!("not_done.shape = {:?}", not_done.size());

            let tgt = {
                let next_q = no_grad(|| {
                    let next_a = self.actor_tgt.forward(&next_o);
                    self.critic_tgt.forward(&next_o, &next_a)
                });
                trace!("         r.size(): {:?}", r.size());
                trace!("  not_done.size(): {:?}", not_done.size());
                trace!("    next_q.size(): {:?}", next_q.size());
                r + not_done * Tensor::from(self.gamma) * next_q
            };
            let pred = self.critic.forward(&o, &a);

            let pred = pred.squeeze();
            let tgt = tgt.squeeze();
            debug_assert_eq!(pred.size().as_slice(), [self.batch_size as i64]);
            debug_assert_eq!(tgt.size().as_slice(), [self.batch_size as i64]);
            trace!("      pred.size(): {:?}", pred.size());
            trace!("       tgt.size(): {:?}", tgt.size());

            // let loss = pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0);
            let loss = pred.mse_loss(&tgt, tch::Reduction::Mean);
            trace!("    critic loss: {:?}", loss);

            loss
        };

        self.critic.backward_step(&loss);

        f32::from(loss)
    }

    fn update_actor(&mut self, batch: &TchBatch<E, O, A>) -> f32 {
        trace!("DDPG.update_actor()");

        let loss = {
            let obs = &batch.obs;
            let act = self.actor.forward(obs);
            let loss = -self.critic.forward(obs, &act).mean(tch::Kind::Float);

            // trace!("  obs.size(): {:?}", obs.size());
            // trace!("    a.size(): {:?}", a.size());
            // trace!("log_p.size(): {:?}", log_p.size());
            // trace!(" qval.size(): {:?}", qval.size());
            trace!("  actor loss: {:?}", loss);

            // let mut stdin = io::stdin();
            // let _ = stdin.read(&mut [0u8]).unwrap();

            loss
        };

        self.actor.backward_step(&loss);

        f32::from(loss)
    }

    fn soft_update(&mut self) {
        track(&mut self.critic_tgt, &mut self.critic, self.tau);
        track(&mut self.actor_tgt, &mut self.actor, self.tau);
    }
}

impl<E, Q, P, O, A> Policy<E> for DDPG<E, Q, P, O, A> where
    E: Env,
    // Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = A::SubBatch> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let act = tch::no_grad(|| self.actor.forward(&obs));
        if self.train {
            self.action_noise.apply(&act).clip(-1.0, 1.0).into()
        }
        else {
            act.into()
        }
    }
}

impl<E, Q, P, O, A> Agent<E> for DDPG<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = A::SubBatch> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
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
    /// * `loss_critic`: Loss of critic
    /// * `loss_actor`: Loss of actor
    fn observe(&mut self, step: Step<E>) -> Option<Record> {
        trace!("DDPG::observe()");

        // Check if doing optimization
        let do_optimize = self.opt_interval_counter.do_optimize(&step.is_done)
            && self.replay_buffer.len() + 1 >= self.min_transitions_warmup;
    
        // Push transition to the replay buffer
        self.push_transition(step);
        trace!("Push transition");

        // Do optimization
        if do_optimize {
            let mut loss_critic = 0f32;
            let mut loss_actor = 0f32;

            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                trace!("Sample random batch");

                loss_critic += self.update_critic(&batch);
                loss_actor += self.update_actor(&batch);
                self.soft_update();
                trace!("Update models");
            };

            loss_critic /= self.n_updates_per_opt as f32;
            loss_actor /= self.n_updates_per_opt as f32;

            Some(Record::from_slice(&[
                ("loss_critic", RecordValue::Scalar(loss_critic)),
                ("loss_actor", RecordValue::Scalar(loss_actor))
            ]))
        }
        else {
            None
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        self.critic.save(&path.as_ref().join("critic.pt").as_path())?;
        self.critic_tgt.save(&path.as_ref().join("critic_tgt.pt").as_path())?;
        self.actor.save(&path.as_ref().join("actor.pt").as_path())?;
        self.actor_tgt.save(&path.as_ref().join("actor_tgt.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.critic.load(&path.as_ref().join("critic.pt").as_path())?;
        self.critic_tgt.load(&path.as_ref().join("critic_tgt.pt").as_path())?;
        self.actor.load(&path.as_ref().join("actor.pt").as_path())?;
        self.actor_tgt.load(&path.as_ref().join("actor_tgt.pt").as_path())?;
        Ok(())
    }
}
