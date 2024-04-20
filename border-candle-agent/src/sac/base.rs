use super::{Actor, Critic, EntCoef, SacConfig};
use crate::{
    model::{SubModel1, SubModel2},
    util::{smooth_l1_loss, track, CriticLoss, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Env, Policy, ReplayBufferBase, StdBatchBase,
};
use candle_core::{Device, Tensor, D};
use candle_nn::loss::mse;
use log::trace;
use serde::{de::DeserializeOwned, Serialize};
use std::{fs, marker::PhantomData, path::Path};

type ActionValue = Tensor;
type ActMean = Tensor;
type ActStd = Tensor;

fn normal_logp(x: &Tensor) -> Result<Tensor> {
    let tmp: Tensor =
        ((-0.5 * (2.0 * std::f32::consts::PI).ln() as f64) - (0.5 * x.powf(2.0)?)?)?;
    Ok(tmp.sum(D::Minus1)?)
}

/// Soft actor critic (SAC) agent.
pub struct Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
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
    pub(super) batch_size: usize,
    pub(super) train: bool,
    pub(super) reward_scale: f32,
    pub(super) n_opts: usize,
    pub(super) critic_loss: CriticLoss,
    pub(super) phantom: PhantomData<(E, R)>,
    pub(super) device: Device,
}

impl<E, Q, P, R> Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
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
    /// Returns action and its log probability under the Normal distributioni.
    fn action_logp(&self, o: &P::Input) -> Result<(Tensor, Tensor)> {
        let (mean, lstd) = self.pi.forward(o);
        let std = lstd.clamp(self.min_lstd, self.max_lstd)?.exp()?;
        let z = Tensor::randn(0f32, 1f32, mean.dims(), &self.device)?;
        let a = (&std * &z + &mean)?.tanh()?;
        let log_p = (normal_logp(&z)?
            - 1f64
                * ((1f64 - a.powf(2.0)?)? + self.epsilon)?
                    .log()?
                    .sum(D::Minus1)?)?
        .squeeze(D::Minus1)?;

        debug_assert_eq!(a.dims()[0], self.batch_size);
        debug_assert_eq!(log_p.dims(), [self.batch_size]);

        Ok((a, log_p))
    }

    fn qvals(&self, qnets: &[Critic<Q>], obs: &Q::Input1, act: &Q::Input2) -> Vec<Tensor> {
        qnets
            .iter()
            .map(|qnet| qnet.forward(obs, act).squeeze(D::Minus1).unwrap())
            .collect()
    }

    /// Returns the minimum values of q values over critics
    fn qvals_min(&self, qnets: &[Critic<Q>], obs: &Q::Input1, act: &Q::Input2) -> Result<Tensor> {
        let qvals = self.qvals(qnets, obs, act);
        let qvals = Tensor::stack(&qvals, 0)?;
        let qvals_min = qvals.min(0)?.squeeze(D::Minus1)?;

        debug_assert_eq!(qvals_min.dims(), [self.batch_size]);

        Ok(qvals_min)
    }

    fn update_critic(&mut self, batch: R::Batch) -> Result<f32> {
        let losses = {
            let (obs, act, next_obs, reward, is_done, _, _) = batch.unpack();
            let batch_size = reward.len();
            let reward = Tensor::from_slice(&reward[..], (batch_size,), &self.device)?;
            let is_done = {
                let is_done = is_done.iter().map(|e| *e as f32).collect::<Vec<_>>();
                Tensor::from_slice(&is_done[..], (batch_size,), &self.device)?
            };

            let preds = self.qvals(&self.qnets, &obs.into(), &act.into());
            let tgt = {
                let next_q = {
                    let (next_a, next_log_p) = self.action_logp(&next_obs.clone().into())?;
                    let next_q =
                        self.qvals_min(&self.qnets_tgt, &next_obs.into(), &next_a.into())?;
                    (next_q - self.ent_coef.alpha()?.broadcast_mul(&next_log_p))?
                };
                ((self.reward_scale as f64) * reward)? + (1f64 - &is_done)? * self.gamma * next_q
            }?
            .detach();

            debug_assert_eq!(tgt.dims(), [self.batch_size]);

            let losses: Vec<_> = match self.critic_loss {
                CriticLoss::Mse => preds
                    .iter()
                    .map(|pred| mse(&pred.squeeze(D::Minus1).unwrap(), &tgt).unwrap())
                    .collect(),
                CriticLoss::SmoothL1 => preds
                    .iter()
                    .map(|pred| smooth_l1_loss(&pred, &tgt).unwrap())
                    .collect(),
            };
            losses
        };

        for (qnet, loss) in self.qnets.iter_mut().zip(&losses) {
            qnet.backward_step(&loss).unwrap();
        }

        Ok(losses
            .iter()
            .map(|loss| loss.to_scalar::<f32>().unwrap())
            .sum::<f32>()
            / (self.qnets.len() as f32))
    }

    fn update_actor(&mut self, batch: &R::Batch) -> Result<f32> {
        let loss = {
            let o = batch.obs().clone();
            let (a, log_p) = self.action_logp(&o.into())?;

            // Update the entropy coefficient
            self.ent_coef.update(&log_p)?;

            let o = batch.obs().clone();
            let qval = self.qvals_min(&self.qnets, &o.into(), &a.into())?;
            ((self.ent_coef.alpha()?.broadcast_mul(&log_p))? - &qval)?.mean_all()?
        };

        self.pi.backward_step(&loss)?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn soft_update(&mut self) -> Result<()> {
        for (qnet_tgt, qnet) in self.qnets_tgt.iter().zip(&mut self.qnets) {
            track(qnet_tgt.get_varmap(), qnet.get_varmap(), self.tau)?;
        }
        Ok(())
    }

    fn opt_(&mut self, buffer: &mut R) -> Result<Record> {
        let mut loss_critic = 0f32;
        let mut loss_actor = 0f32;

        for _ in 0..self.n_updates_per_opt {
            trace!("batch()");
            let batch = buffer.batch(self.batch_size).unwrap();

            trace!("update_actor()");
            loss_actor += self.update_actor(&batch)?;

            trace!("update_critic()");
            loss_critic += self.update_critic(batch)?;

            trace!("soft_update()");
            self.soft_update()?;

            self.n_opts += 1;
        }

        loss_critic /= self.n_updates_per_opt as f32;
        loss_actor /= self.n_updates_per_opt as f32;

        let record = Record::from_slice(&[
            ("loss_critic", RecordValue::Scalar(loss_critic)),
            ("loss_actor", RecordValue::Scalar(loss_actor)),
            (
                "ent_coef",
                RecordValue::Scalar(self.ent_coef.alpha()?.to_vec1::<f32>()?[0]),
            ),
        ]);

        Ok(record)
    }
}

impl<E, Q, P, R> Policy<E> for Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
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
        let device = config.device.expect("No device is given for SAC agent");
        let n_critics = config.n_critics;
        let pi = Actor::build(config.actor_config, device.clone().into()).unwrap();
        let mut qnets = vec![];
        let mut qnets_tgt = vec![];
        for _ in 0..n_critics {
            qnets.push(Critic::build(config.critic_config.clone(), device.clone().into()).unwrap());
            qnets_tgt
                .push(Critic::build(config.critic_config.clone(), device.clone().into()).unwrap());
        }

        // if let Some(seed) = config.seed.as_ref() {
        //     tch::manual_seed(*seed);
        // }

        Sac {
            qnets,
            qnets_tgt,
            pi,
            gamma: config.gamma,
            tau: config.tau,
            ent_coef: EntCoef::new(config.ent_coef_mode, device.into()).unwrap(),
            epsilon: config.epsilon,
            min_lstd: config.min_lstd,
            max_lstd: config.max_lstd,
            n_updates_per_opt: config.n_updates_per_opt,
            batch_size: config.batch_size,
            train: config.train,
            reward_scale: config.reward_scale,
            critic_loss: config.critic_loss,
            n_opts: 0,
            device: device.into(),
            phantom: PhantomData,
        }
    }

    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let (mean, lstd) = self.pi.forward(&obs);
        let std = lstd
            .clamp(self.min_lstd, self.max_lstd)
            .unwrap()
            .exp()
            .unwrap();
        let act = if self.train {
            ((std * mean.randn_like(0., 1.).unwrap()).unwrap() + mean).unwrap()
        } else {
            mean
        };
        act.tanh().unwrap().into()
    }
}

impl<E, Q, P, R> Agent<E, R> for Sac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
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

    fn opt_with_record(&mut self, buffer: &mut R) -> Record {
        self.opt_(buffer).expect("Failed in Sac::opt_()")
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
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

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
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

// #[cfg(feature = "border-async-trainer")]
// use {crate::util::NamedTensors, border_async_trainer::SyncModel};

// #[cfg(feature = "border-async-trainer")]
// impl<E, Q, P, R> SyncModel for Sac<E, Q, P, R>
// where
//     E: Env,
//     Q: SubModel2<Output = ActionValue>,
//     P: SubModel<Output = (ActMean, ActStd)>,
//     R: ReplayBufferBase,
//     E::Obs: Into<Q::Input1> + Into<P::Input>,
//     E::Act: Into<Q::Input2> + From<Tensor>,
//     Q::Input2: From<ActMean>,
//     Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
//     P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
//     R::Batch: StdBatchBase,
//     <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
//     <R::Batch as StdBatchBase>::ActBatch: Into<Q::Input2> + Into<Tensor>,
// {
//     type ModelInfo = NamedTensors;

//     fn model_info(&self) -> (usize, Self::ModelInfo) {
//         (
//             self.n_opts,
//             NamedTensors::copy_from(self.pi.get_var_store()),
//         )
//     }

//     fn sync_model(&mut self, model_info: &Self::ModelInfo) {
//         model_info.copy_to(self.pi.get_var_store_mut());
//     }
// }
