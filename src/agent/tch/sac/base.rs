//! SAC agent.
use crate::agent::{
    tch::{
        model::{Model1, Model2},
        sac::ent_coef::EntCoef,
        util::track,
        ReplayBuffer, TchBatch, TchBuffer,
    },
    CriticLoss, OptIntervalCounter,
};
use border_core::{
    record::{Record, RecordValue},
    Agent, Env, Policy, Step,
};
use log::trace;
use std::{cell::RefCell, error::Error, fs, marker::PhantomData, path::Path};
use tch::{no_grad, Tensor};

type ActionValue = Tensor;
type ActMean = Tensor;
type ActStd = Tensor;

fn normal_logp(x: &Tensor) -> Tensor {
    let tmp: Tensor =
        Tensor::from(-0.5 * (2.0 * std::f32::consts::PI).ln() as f32) - 0.5 * x.pow(2);
    tmp.sum1(&[-1], false, tch::Kind::Float)
}

/// Soft actor critic agent.
#[allow(clippy::upper_case_acronyms)]
pub struct SAC<E, Q, P, O, A>
where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    pub(in crate::agent::tch::sac) qnets: Vec<Q>,
    pub(in crate::agent::tch::sac) qnets_tgt: Vec<Q>,
    pub(in crate::agent::tch::sac) pi: P,
    pub(in crate::agent::tch::sac) replay_buffer: ReplayBuffer<E, O, A>,
    pub(in crate::agent::tch::sac) gamma: f64,
    pub(in crate::agent::tch::sac) tau: f64,
    pub(in crate::agent::tch::sac) ent_coef: EntCoef,
    pub(in crate::agent::tch::sac) epsilon: f64,
    pub(in crate::agent::tch::sac) min_lstd: f64,
    pub(in crate::agent::tch::sac) max_lstd: f64,
    pub(in crate::agent::tch::sac) opt_interval_counter: OptIntervalCounter,
    pub(in crate::agent::tch::sac) n_updates_per_opt: usize,
    pub(in crate::agent::tch::sac) min_transitions_warmup: usize,
    pub(in crate::agent::tch::sac) batch_size: usize,
    pub(in crate::agent::tch::sac) train: bool,
    pub(in crate::agent::tch::sac) reward_scale: f32,
    pub(in crate::agent::tch::sac) critic_loss: CriticLoss,
    pub(in crate::agent::tch::sac) prev_obs: RefCell<Option<E::Obs>>,
    pub(in crate::agent::tch::sac) phantom: PhantomData<E>,
    pub(in crate::agent::tch::sac) device: tch::Device,
}

impl<E, Q, P, O, A> SAC<E, Q, P, O, A>
where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Input = Tensor, Output = (ActMean, ActStd)> + Clone,
    E::Obs: Into<O::SubBatch>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    // Adapted from dqn.rs
    fn push_transition(&mut self, step: Step<E>) {
        let next_obs = step.obs;
        let obs = self.prev_obs.replace(None).unwrap();
        let reward = Tensor::of_slice(&step.reward[..]);
        let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);
        self.replay_buffer
            .push(&obs, &step.act, &reward, &next_obs, &not_done);
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn action_logp(&self, o: &P::Input) -> (A::SubBatch, Tensor) {
        trace!("SAC::action_logp()");

        let (mean, lstd) = self.pi.forward(o);
        let std = lstd.clip(self.min_lstd, self.max_lstd).exp();
        let z = Tensor::randn(mean.size().as_slice(), tch::kind::FLOAT_CPU).to(self.device);
        let a = (&std * &z + &mean).tanh();
        let log_p = normal_logp(&z)
            - (Tensor::from(1f32) - a.pow(2.0) + Tensor::from(self.epsilon))
                .log()
                .sum1(&[-1], false, tch::Kind::Float);

        debug_assert_eq!(a.size().as_slice()[0], self.batch_size as i64);
        debug_assert_eq!(log_p.size().as_slice(), [self.batch_size as i64]);

        (a, log_p)
    }

    fn qvals(&self, qnets: &[Q], obs: &Tensor, act: &Tensor) -> Vec<Tensor> {
        qnets
            .iter()
            .map(|qnet| qnet.forward(obs, act).squeeze())
            .collect()
    }

    /// Returns the minimum values of q values over critics
    fn qvals_min(&self, qnets: &[Q], obs: &Tensor, act: &Tensor) -> Tensor {
        let qvals = self.qvals(qnets, obs, act);
        let qvals = Tensor::vstack(&qvals);
        let qvals_min = qvals.min2(0, false).0;

        debug_assert_eq!(qvals_min.size().as_slice(), [self.batch_size as i64]);

        qvals_min
    }

    // /// Returns the minimum values of q values over critics
    // fn qvals_mean(&self, qnets: &[Q], obs: &Tensor, act: &Tensor) -> Tensor {
    //     let qvals = self.qvals(qnets, obs, act);
    //     let qvals = Tensor::vstack(&qvals);
    //     let qvals_mean = qvals.mean1(&[0], false, tch::Kind::Float);

    //     debug_assert_eq!(qvals_mean.size().as_slice(), [self.batch_size as i64]);

    //     qvals_mean
    // }

    fn update_critic(&mut self, batch: &TchBatch<E, O, A>) -> f32 {
        trace!("SAC::update_critic()");

        let losses = {
            let o = &batch.obs.to(self.device);
            let a = &batch.actions.to(self.device);
            let next_o = &batch.next_obs.to(self.device);
            let r = &batch.rewards.to(self.device).squeeze();
            let not_done = &batch.not_dones.to(self.device).squeeze();

            let preds = self.qvals(&self.qnets, &o, &a);
            let tgt = {
                let next_q = no_grad(|| {
                    let (next_a, next_log_p) = self.action_logp(&next_o);
                    let next_q = self.qvals_min(&self.qnets_tgt, &next_o, &next_a);
                    next_q - self.ent_coef.alpha() * next_log_p
                });
                self.reward_scale * r + not_done * Tensor::from(self.gamma) * next_q
            };

            debug_assert_eq!(tgt.size().as_slice(), [self.batch_size as i64]);

            let losses: Vec<_> = match self.critic_loss {
                CriticLoss::MSE => preds
                    .iter()
                    .map(|pred| pred.mse_loss(&tgt, tch::Reduction::Mean))
                    .collect(),
                CriticLoss::SmoothL1 => preds
                    .iter()
                    .map(|pred| pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0))
                    .collect(),
            };

            losses
        };

        for (qnet, loss) in self.qnets.iter_mut().zip(&losses) {
            qnet.backward_step(&loss);
        }

        losses.iter().map(f32::from).sum::<f32>() / (self.qnets.len() as f32)
    }

    fn update_actor(&mut self, batch: &TchBatch<E, O, A>) -> f32 {
        trace!("SAC::update_actor()");

        let loss = {
            let o = &batch.obs.to(self.device);
            let (a, log_p) = self.action_logp(o);

            // Update the entropy coefficient
            self.ent_coef.update(&log_p);

            // let qval = self.qvals_mean(&self.qnets, o, &a);
            let qval = self.qvals_min(&self.qnets, o, &a);
            (self.ent_coef.alpha() * &log_p - &qval).mean(tch::Kind::Float)
        };

        self.pi.backward_step(&loss);

        f32::from(loss)
    }

    fn soft_update(&mut self) {
        for (qnet_tgt, qnet) in self.qnets_tgt.iter_mut().zip(&mut self.qnets) {
            track(qnet_tgt, qnet, self.tau);
        }
    }
}

impl<E, Q, P, O, A> Policy<E> for SAC<E, Q, P, O, A>
where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Input = Tensor, Output = (ActMean, ActStd)> + Clone,
    E::Obs: Into<O::SubBatch>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into().to(self.device);
        let (mean, lstd) = self.pi.forward(&obs);
        let std = lstd.clip(self.min_lstd, self.max_lstd).exp();
        let act = if self.train {
            std * Tensor::randn(&mean.size(), tch::kind::FLOAT_CPU).to(self.device) + mean
        } else {
            mean
        };
        act.tanh().into()
    }
}

impl<E, Q, P, O, A> Agent<E> for SAC<E, Q, P, O, A>
where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Input = Tensor, Output = (ActMean, ActStd)> + Clone,
    E::Obs: Into<O::SubBatch>,
    E::Act: From<Tensor>,
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
        trace!("SAC::observe()");

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

                loss_critic += self.update_critic(&batch);
                loss_actor += self.update_actor(&batch);
                self.soft_update();
            }

            Some(Record::from_slice(&[
                ("loss_critic", RecordValue::Scalar(loss_critic)),
                ("loss_actor", RecordValue::Scalar(loss_actor)),
                (
                    "ent_coef",
                    RecordValue::Scalar(self.ent_coef.alpha().double_value(&[0]) as f32),
                ),
            ]))
        } else {
            None
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        for (i, (qnet, qnet_tgt)) in self.qnets.iter().zip(&self.qnets_tgt).enumerate() {
            qnet.save(&path.as_ref().join(format!("qnet_{}.pt", i)).as_path())?;
            qnet_tgt.save(&path.as_ref().join(format!("qnet_tgt_{}.pt", i)).as_path())?;
        }
        self.pi.save(&path.as_ref().join("pi.pt").as_path())?;
        self.ent_coef
            .save(&path.as_ref().join("ent_coef.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        for (i, (qnet, qnet_tgt)) in self.qnets.iter_mut().zip(&mut self.qnets_tgt).enumerate() {
            qnet.load(&path.as_ref().join(format!("qnet_{}.pt", i)).as_path())?;
            qnet_tgt.load(&path.as_ref().join(format!("qnet_tgt_{}.pt", i)).as_path())?;
        }
        self.pi.load(&path.as_ref().join("pi.pt").as_path())?;
        self.ent_coef
            .load(&path.as_ref().join("ent_coef.pt").as_path())?;
        Ok(())
    }
}
