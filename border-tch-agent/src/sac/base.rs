use super::{Actor, Critic, EntCoef, SacConfig};
use crate::{
    model::{ModelBase, SubModel, SubModel2},
    util::{track, CriticLoss, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Env, Policy, ReplayBufferBase, StdBatchBase,
};
use serde::{de::DeserializeOwned, Serialize};
// use log::info;
use std::{fs, marker::PhantomData, path::Path};
use tch::{no_grad, Tensor};

type ActionValue = Tensor;
type ActMean = Tensor;
type ActStd = Tensor;

fn normal_logp(x: &Tensor) -> Tensor {
    let tmp: Tensor = Tensor::from(-0.5 * (2.0 * std::f32::consts::PI).ln() as f32)
        - 0.5 * x.pow_tensor_scalar(2);
    tmp.sum_dim_intlist(Some([-1].as_slice()), false, tch::Kind::Float)
}

/// Soft actor critic (SAC) agent.
pub struct Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel<Output = (ActMean, ActStd)>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2>,
    Q::Input2: From<ActMean>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as StdBatchBase>::ActBatch: Into<Q::Input2> + Into<Tensor>,
{
    pub(super) qnets: Vec<Critic<Q>>,
    pub(super) qnets_tgt: Vec<Critic<Q>>,
    pub(super) pi: Actor<P>,
    pub(super) gamma: f64,
    pub(super) tau: f64,
    pub(super) ent_coef: EntCoef,
    pub(super) epsilon: f64,
    pub(super) min_lstd: f64,
    pub(super) max_lstd: f64,
    pub(super) n_updates_per_opt: usize,
    pub(super) min_transitions_warmup: usize,
    pub(super) batch_size: usize,
    pub(super) train: bool,
    pub(super) reward_scale: f32,
    pub(super) n_opts: usize,
    pub(super) critic_loss: CriticLoss,
    pub(super) phantom: PhantomData<(E, R)>,
    pub(super) device: tch::Device,
}

impl<E, Q, P, R> Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel<Output = (ActMean, ActStd)>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2>,
    Q::Input2: From<ActMean>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as StdBatchBase>::ActBatch: Into<Q::Input2> + Into<Tensor>,
{
    fn action_logp(&self, o: &P::Input) -> (Tensor, Tensor) {
        let (mean, lstd) = self.pi.forward(o);
        let std = lstd.clip(self.min_lstd, self.max_lstd).exp();
        let z = Tensor::randn(mean.size().as_slice(), tch::kind::FLOAT_CPU).to(self.device);
        let a = (&std * &z + &mean).tanh();
        let log_p = normal_logp(&z)
            - (Tensor::from(1f32) - a.pow_tensor_scalar(2.0) + Tensor::from(self.epsilon))
                .log()
                .sum_dim_intlist(Some([-1].as_slice()), false, tch::Kind::Float);

        debug_assert_eq!(a.size().as_slice()[0], self.batch_size as i64);
        debug_assert_eq!(log_p.size().as_slice(), [self.batch_size as i64]);

        (a, log_p)
    }

    fn qvals(&self, qnets: &[Critic<Q>], obs: &Q::Input1, act: &Q::Input2) -> Vec<Tensor> {
        qnets
            .iter()
            .map(|qnet| qnet.forward(obs, act).squeeze())
            .collect()
    }

    /// Returns the minimum values of q values over critics
    fn qvals_min(&self, qnets: &[Critic<Q>], obs: &Q::Input1, act: &Q::Input2) -> Tensor {
        let qvals = self.qvals(qnets, obs, act);
        let qvals = Tensor::vstack(&qvals);
        let qvals_min = qvals.min_dim(0, false).0;

        debug_assert_eq!(qvals_min.size().as_slice(), [self.batch_size as i64]);

        qvals_min
    }

    fn update_critic(&mut self, batch: R::Batch) -> f32 {
        let losses = {
            let (obs, act, next_obs, reward, is_done, _, _) = batch.unpack();
            let reward = Tensor::of_slice(&reward[..]).to(self.device);
            let is_done = Tensor::of_slice(&is_done[..]).to(self.device);

            let preds = self.qvals(&self.qnets, &obs.into(), &act.into());
            let tgt = {
                let next_q = no_grad(|| {
                    let (next_a, next_log_p) = self.action_logp(&next_obs.clone().into());
                    let next_q = self.qvals_min(&self.qnets_tgt, &next_obs.into(), &next_a.into());
                    next_q - self.ent_coef.alpha() * next_log_p
                });
                self.reward_scale * reward + (1f32 - &is_done) * Tensor::from(self.gamma) * next_q
            };

            debug_assert_eq!(tgt.size().as_slice(), [self.batch_size as i64]);

            let losses: Vec<_> = match self.critic_loss {
                CriticLoss::Mse => preds
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

    fn update_actor(&mut self, batch: &R::Batch) -> f32 {
        let loss = {
            let o = batch.obs().clone();
            let (a, log_p) = self.action_logp(&o.into());

            // Update the entropy coefficient
            self.ent_coef.update(&log_p);

            let o = batch.obs().clone();
            let qval = self.qvals_min(&self.qnets, &o.into(), &a.into());
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

    fn opt_(&mut self, buffer: &mut R) -> Record {
        let mut loss_critic = 0f32;
        let mut loss_actor = 0f32;

        for _ in 0..self.n_updates_per_opt {
            let batch = buffer.batch(self.batch_size).unwrap();
            loss_actor += self.update_actor(&batch);
            loss_critic += self.update_critic(batch);
            self.soft_update();
            self.n_opts += 1;
        }

        loss_critic /= self.n_updates_per_opt as f32;
        loss_actor /= self.n_updates_per_opt as f32;

        Record::from_slice(&[
            ("loss_critic", RecordValue::Scalar(loss_critic)),
            ("loss_actor", RecordValue::Scalar(loss_actor)),
            (
                "ent_coef",
                RecordValue::Scalar(self.ent_coef.alpha().double_value(&[0]) as f32),
            ),
        ])
    }
}

impl<E, Q, P, R> Policy<E> for Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel<Output = (ActMean, ActStd)>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Input2: From<ActMean>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as StdBatchBase>::ActBatch: Into<Q::Input2> + Into<Tensor>,
{
    type Config = SacConfig<Q, P>;

    /// Constructs [Sac] agent.
    fn build(config: Self::Config) -> Self {
        let device = config
            .device
            .expect("No device is given for SAC agent")
            .into();
        let n_critics = config.n_critics;
        let pi = Actor::build(config.actor_config, device).unwrap();
        let mut qnets = vec![];
        let mut qnets_tgt = vec![];
        for _ in 0..n_critics {
            let critic = Critic::build(config.critic_config.clone(), device).unwrap();
            qnets.push(critic.clone());
            qnets_tgt.push(critic);
        }

        if let Some(seed) = config.seed.as_ref() {
            tch::manual_seed(*seed);
        }

        Sac {
            qnets,
            qnets_tgt,
            pi,
            gamma: config.gamma,
            tau: config.tau,
            ent_coef: EntCoef::new(config.ent_coef_mode, device),
            epsilon: config.epsilon,
            min_lstd: config.min_lstd,
            max_lstd: config.max_lstd,
            n_updates_per_opt: config.n_updates_per_opt,
            min_transitions_warmup: config.min_transitions_warmup,
            batch_size: config.batch_size,
            train: config.train,
            reward_scale: config.reward_scale,
            critic_loss: config.critic_loss,
            n_opts: 0,
            device,
            phantom: PhantomData,
        }
    }

    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
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

impl<E, Q, P, R> Agent<E, R> for Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel<Output = (ActMean, ActStd)>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Input2: From<ActMean>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as StdBatchBase>::ActBatch: Into<Q::Input2> + Into<Tensor>,
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

    fn opt(&mut self, buffer: &mut R) -> Option<Record> {
        if buffer.len() >= self.min_transitions_warmup {
            Some(self.opt_(buffer))
        } else {
            None
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        for (i, (qnet, qnet_tgt)) in self.qnets.iter().zip(&self.qnets_tgt).enumerate() {
            qnet.save(&path.as_ref().join(format!("qnet_{}.pt.tch", i)).as_path())?;
            qnet_tgt.save(&path.as_ref().join(format!("qnet_tgt_{}.pt.tch", i)).as_path())?;
        }
        self.pi.save(&path.as_ref().join("pi.pt.tch").as_path())?;
        self.ent_coef
            .save(&path.as_ref().join("ent_coef.pt.tch").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        for (i, (qnet, qnet_tgt)) in self.qnets.iter_mut().zip(&mut self.qnets_tgt).enumerate() {
            qnet.load(&path.as_ref().join(format!("qnet_{}.pt.tch", i)).as_path())?;
            qnet_tgt.load(&path.as_ref().join(format!("qnet_tgt_{}.pt.tch", i)).as_path())?;
        }
        self.pi.load(&path.as_ref().join("pi.pt.tch").as_path())?;
        self.ent_coef
            .load(&path.as_ref().join("ent_coef.pt.tch").as_path())?;
        Ok(())
    }
}

#[cfg(feature = "border-async-trainer")]
use {crate::util::NamedTensors, border_async_trainer::SyncModel};

#[cfg(feature = "border-async-trainer")]
impl<E, Q, P, R> SyncModel for Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel<Output = (ActMean, ActStd)>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Input2: From<ActMean>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as StdBatchBase>::ActBatch: Into<Q::Input2> + Into<Tensor>,
{
    type ModelInfo = NamedTensors;

    fn model_info(&self) -> (usize, Self::ModelInfo) {
        (
            self.n_opts,
            NamedTensors::copy_from(self.pi.get_var_store()),
        )
    }

    fn sync_model(&mut self, model_info: &Self::ModelInfo) {
        model_info.copy_to(self.pi.get_var_store_mut());
    }
}
