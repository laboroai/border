use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{no_grad, Tensor};
use crate::{core::{Policy, Agent, Step, Env}};
use crate::agents::tch::{ReplayBuffer, TchBuffer, TchBatch};
use crate::agents::tch::model::{Model1, Model2};
use crate::agents::tch::util::{track, sum_keep1};

type ActionValue = Tensor;
type ActMean = Tensor;
type ActStd = Tensor;

fn normal_logp(x: &Tensor, mean: &Tensor, std: &Tensor) -> Tensor {
    Tensor::from(-0.5 * (2.0 * std::f32::consts::PI).ln())
    - 0.5 * ((x - mean) / std).pow(2) - std.log()
}

pub struct SAC<E, Q, P, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    qnet: Q,
    qnet_tgt: Q,
    pi: P,
    replay_buffer: ReplayBuffer<E, O, A>,
    gamma: f64,
    tau: f64,
    alpha: f64,
    epsilon: f64,
    n_samples_per_opt: usize,
    n_updates_per_opt: usize,
    n_opts_per_soft_update: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    count_samples_per_opt: usize,
    count_opts_per_soft_update: usize,
    train: bool,
    prev_obs: RefCell<Option<E::Obs>>,
    phantom: PhantomData<E>
}

impl<E, Q, P, O, A> SAC<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = (ActMean, ActStd)> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
    P::Input: Copy,
    {
    pub fn new(qnet: Q, pi: P, replay_buffer: ReplayBuffer<E, O, A>) -> Self {
        let qnet_tgt = qnet.clone();
        SAC {
            qnet,
            qnet_tgt,
            pi,
            replay_buffer,
            gamma: 0.99,
            tau: 0.005,
            alpha: 1.0,
            epsilon: 1e-8,
            n_samples_per_opt: 1,
            n_updates_per_opt: 1,
            n_opts_per_soft_update: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
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
        self.gamma = v;
        self
    }

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

    fn action_logp(&self, o: &P::Input) -> (A::SubBatch, Tensor) {
        let (mean, std) = self.pi.forward(o);
        let z = Tensor::randn(mean.size().as_slice(), tch::kind::FLOAT_CPU);
        let a = (&std * &z + &mean).tanh();
        let log_p = normal_logp(&z, &mean, &std)
            - (1 - a.pow(2).log() + Tensor::from(self.epsilon));
        let log_p = sum_keep1(&log_p);

        debug_assert!(a.size().as_slice() == [self.batch_size as i64]);
        debug_assert!(log_p.size().as_slice() == [self.batch_size as i64]);

        (a, log_p)
    }

    fn update_critic(&mut self, batch: &TchBatch<E, O, A>) {
        trace!("Start sac.update_critic()");

        let loss = {
            let o_ = batch.obs.clone();
            let a_ = batch.actions.copy();
            let next_o = batch.next_obs.clone();
            let r = batch.rewards.copy();
            let not_done = batch.not_dones.copy();
            // trace!("obs.shape      = {:?}", o.size());
            // trace!("next_obs.shape = {:?}", next_o.size());
            // trace!("act.shape      = {:?}", a.size());
            trace!("reward.shape   = {:?}", r.size());
            trace!("not_done.shape = {:?}", not_done.size());

            let o= &o_;
            let a = &a_;

            let pred = self.qnet.forward(&o, &a);
            let tgt = {
                let next_q = no_grad(|| {
                    let (next_a, next_log_p) = self.action_logp(&next_o);
                    let next_q = self.qnet_tgt.forward(&next_o, &next_a);
                    next_q - self.alpha * next_log_p
                });
                &r + &not_done * Tensor::from(self.gamma) * next_q
            };

            debug_assert!(pred.size().as_slice() == [self.batch_size as i64]);
            debug_assert!(tgt.size().as_slice() == [self.batch_size as i64]);

            0.5 * pred.mse_loss(&tgt, tch::Reduction::Mean)
        };

        self.qnet.backward_step(&loss);
    }

    fn update_actor(&mut self, batch: &TchBatch<E, O, A>) {
        trace!("Start sac.update_actor()");

        let loss = {
            let o = batch.obs.clone();
            let (a, log_p) = self.action_logp(&o);
            self.alpha * log_p.detach() - self.qnet.forward(&o, &a)
        };

        self.pi.backward_step(&loss);
    }

    fn soft_update_qnet_tgt(&mut self) {
        track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
    }
}

impl<E, Q, P, O, A> Policy<E> for SAC<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = (ActMean, ActStd)> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
    P::Input: Copy,
{
    fn sample(&self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let (m, s) = self.pi.forward(&obs);
        let act = if self.train {
            s * Tensor::randn(&m.size(), tch::kind::FLOAT_CPU) + m
        }
        else {
            m
        };
        act.tanh().into()
    }
}

impl<E, Q, P, O, A> Agent<E> for SAC<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = (ActMean, ActStd)> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
    P::Input: Copy,
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
        trace!("Start dqn.observe()");

        // Push transition to the replay buffer
        self.push_transition(step);
        trace!("Push transition");

        // Do optimization 1 step
        self.count_samples_per_opt += 1;
        if self.count_samples_per_opt == self.n_samples_per_opt {
            self.count_samples_per_opt = 0;

            if self.replay_buffer.len() >= self.min_transitions_warmup {
                for _ in 0..self.n_updates_per_opt {
                    let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                    trace!("Sample random batch");

                    self.update_critic(&batch);
                    self.update_actor(&batch);
                    trace!("Update models");
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
