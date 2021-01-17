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
    Tensor::from(-0.5 * (2.0 * std::f32::consts::PI).ln() as f32)
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
    max_std: f64,
    n_samples_per_opt: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    count_samples_per_opt: usize,
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
            alpha: 0.1,
            epsilon: 1e-8,
            max_std: 100.0,
            n_samples_per_opt: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            count_samples_per_opt: 0,
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

    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
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
        trace!("SAC.action_logp()");

        let (mean, lstd) = self.pi.forward(o);
        let std = lstd.exp().minimum(&Tensor::from(self.max_std));
        let z = Tensor::randn(mean.size().as_slice(), tch::kind::FLOAT_CPU);
        // TODO: parametrize output scale; 2.0 is for pendulum env
        let a = (&std * &z + &mean).tanh();
        let log_p = normal_logp(&z, &mean, &std)
            - (Tensor::from(1f32) - a.pow(2.0) + Tensor::from(self.epsilon)).log();
        let log_p = sum_keep1(&log_p);

        trace!(" mean.size(): {:?}", mean.size());
        trace!("  std.size(): {:?}", std.size());
        trace!("    z.size(): {:?}", z.size());
        trace!("log_p.size(): {:?}", log_p.size());

        debug_assert_eq!(a.size().as_slice()[0], self.batch_size as i64);
        debug_assert_eq!(log_p.size().as_slice(), [self.batch_size as i64]);

        (a, log_p)
    }

    fn update_critic(&mut self, batch: &TchBatch<E, O, A>) {
        trace!("SAC.update_critic()");

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

            let pred = self.qnet.forward(&o, &a);
            let tgt = {
                let next_q = no_grad(|| {
                    let (next_a, next_log_p) = self.action_logp(&next_o);
                    let next_q = self.qnet_tgt.forward(&next_o, &next_a);
                    trace!("    next_q.size(): {:?}", next_q.size());
                    trace!("next_log_p.size(): {:?}", next_log_p.size());
                    next_q - self.alpha * (next_log_p.unsqueeze(-1))
                });
                trace!("         r.size(): {:?}", r.size());
                trace!("  not_done.size(): {:?}", not_done.size());
                trace!("    next_q.size(): {:?}", next_q.size());
                r + not_done * Tensor::from(self.gamma) * next_q
            };

            let pred = pred.squeeze();
            let tgt = tgt.squeeze();
            debug_assert_eq!(pred.size().as_slice(), [self.batch_size as i64]);
            debug_assert_eq!(tgt.size().as_slice(), [self.batch_size as i64]);
            trace!("      pred.size(): {:?}", pred.size());
            trace!("       tgt.size(): {:?}", tgt.size());

            let loss = pred.mse_loss(&tgt, tch::Reduction::Mean);
            trace!("    critic loss: {:?}", loss);

            loss
        };

        self.qnet.backward_step(&loss);
    }

    fn update_actor(&mut self, batch: &TchBatch<E, O, A>) {
        trace!("SAC.update_actor()");

        let loss = {
            let o = &batch.obs;
            let (a, log_p) = self.action_logp(o);
            let qval = self.qnet.forward(o, &a).squeeze();
            let loss = (self.alpha * &log_p - &qval).mean(tch::Kind::Float);

            trace!("    a.size(): {:?}", a.size());
            trace!("log_p.size(): {:?}", log_p.size());
            trace!(" qval.size(): {:?}", qval.size());
            trace!("  actor loss: {:?}", loss);

            // let mut stdin = io::stdin();
            // let _ = stdin.read(&mut [0u8]).unwrap();

            loss
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
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let (mean, lstd) = self.pi.forward(&obs);
        let std = lstd.exp().minimum(&Tensor::from(self.max_std));
        let act = if self.train {
            std * Tensor::randn(&mean.size(), tch::kind::FLOAT_CPU) + mean
        }
        else {
            mean
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
                    self.soft_update_qnet_tgt();
                    trace!("Update models");
                };
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
