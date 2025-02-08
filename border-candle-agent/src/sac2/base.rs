use super::{EntCoef, Sac2Config};
use crate::{
    model::{SubModel1, SubModel2},
    util::{
        actor::GaussianActor, critic::MultiCritic, gamma_not_done, smooth_l1_loss, CriticLoss,
        OutDim,
    },
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Configurable, Env, Policy, ReplayBufferBase, TransitionBatch,
};
use candle_core::{Device, Tensor, D};
use candle_nn::loss::mse;
use log::trace;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs,
    marker::PhantomData,
    path::{Path, PathBuf},
};

type ActionValue = Tensor;
type ActMean = Tensor;
type ActStd = Tensor;

/// Soft actor critic (SAC) agent.
pub struct Sac2<E, Q, P, R>
where
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    critic: MultiCritic<Q>,
    actor: GaussianActor<P>,
    gamma: f64,
    ent_coef: EntCoef,
    n_updates_per_opt: usize,
    batch_size: usize,
    train: bool,
    n_opts: usize,
    critic_loss: CriticLoss,
    phantom: PhantomData<(E, R)>,
    device: Device,
}

impl<E, Q, P, R> Sac2<E, Q, P, R>
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
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as TransitionBatch>::ActBatch: Into<Q::Input2> + Into<Tensor>,
{
    fn update_critic(&mut self, batch: R::Batch) -> Result<f32> {
        let loss = {
            // Extract items in the batch
            let (obs, act, next_obs, reward, is_terminated, is_truncated, _, _) = batch.unpack();
            let batch_size = reward.len();
            let reward = Tensor::from_slice(&reward[..], (batch_size,), &self.device)?;

            // Prediction
            let qs = self.critic.qvals(&obs.into(), &act.into());

            // Target
            let tgt = {
                let gamma_not_done =
                    gamma_not_done(self.gamma as f32, is_terminated, is_truncated, &self.device)?;
                let next_act = self.actor.sample(&next_obs.clone().into(), self.train)?;
                let next_log_p = self.actor.logp(&next_obs.clone().into(), &next_act)?;
                let next_q = self
                    .critic
                    .qvals_min_tgt(&next_obs.into(), &next_act.into())?;
                let next_q = next_q - self.ent_coef.alpha()?.broadcast_mul(&next_log_p)?;
                (&reward + (&gamma_not_done * next_q)?)?.squeeze(D::Minus1)?
            }
            .detach();
            debug_assert_eq!(tgt.dims(), [self.batch_size]);

            // Loss
            let losses: Vec<_> = match self.critic_loss {
                CriticLoss::Mse => qs.iter().map(|pred| mse(&pred, &tgt).unwrap()).collect(),
                CriticLoss::SmoothL1 => qs
                    .iter()
                    .map(|pred| smooth_l1_loss(&pred, &tgt).unwrap())
                    .collect(),
            };
            Tensor::stack(&losses, 0)?.mean_all()?
        };

        self.critic.backward_step(&loss)?;
        // self.critic.soft_update()?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn update_actor(&mut self, batch: &R::Batch) -> Result<f32> {
        let loss = {
            let obs = batch.obs().clone();
            let act = self.actor.sample(&obs.clone().into(), self.train)?;
            let log_p = self.actor.logp(&obs.clone().into(), &act)?;

            // Update the entropy coefficient
            self.ent_coef.update(&log_p.detach())?;

            // Loss
            let q = self.critic.qvals_min(&obs.into(), &act.into())?;
            let alpha = self.ent_coef.alpha()?.detach();
            (alpha.broadcast_mul(&log_p)? - &q)?.mean_all()?
        };

        self.actor.backward_step(&loss)?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn opt_(&mut self, buffer: &mut R) -> Result<Record> {
        let mut loss_critic = 0f32;
        let mut loss_actor = 0f32;

        for _ in 0..self.n_updates_per_opt {
            let batch = buffer.batch(self.batch_size).unwrap();
            loss_actor += 0.0; //self.update_actor(&batch)?;
            loss_critic += self.update_critic(batch)?;
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

impl<E, Q, P, R> Policy<E> for Sac2<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        self.actor
            .sample(&obs.clone().into(), self.train)
            .unwrap()
            .into()
    }
}

impl<E, Q, P, R> Configurable for Sac2<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    type Config = Sac2Config<Q, P>;

    /// Constructs [`Sac2`] agent.
    fn build(config: Self::Config) -> Self {
        let device: Device = config
            .device
            .expect("No device is given for AWAC agent")
            .into();
        // let device = config.device.expect("No device is given for SAC agent");
        let actor = GaussianActor::build(config.actor_config, device.clone().into()).unwrap();
        let critic = MultiCritic::build(config.critic_config, device.clone().into()).unwrap();
        let ent_coef = EntCoef::new(config.ent_coef_mode, device.clone().into()).unwrap();

        // if let Some(seed) = config.seed.as_ref() {
        //     tch::manual_seed(*seed);
        // }

        Sac2 {
            actor,
            critic,
            gamma: config.gamma,
            ent_coef,
            n_updates_per_opt: config.n_updates_per_opt,
            batch_size: config.batch_size,
            train: false,
            critic_loss: config.critic_loss,
            n_opts: 0,
            device: device.into(),
            phantom: PhantomData,
        }
    }
}

impl<E, Q, P, R> Agent<E, R> for Sac2<E, Q, P, R>
where
    E: Env + 'static,
    Q: SubModel2<Output = ActionValue> + 'static,
    P: SubModel1<Output = (ActMean, ActStd)> + 'static,
    R: ReplayBufferBase + 'static,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Input2: From<ActMean>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as TransitionBatch>::ActBatch: Into<Q::Input2> + Into<Tensor>,
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

    fn save_params(&self, path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;

        let actor_path = self.actor.save(path.join("actor"))?;
        let (critic_path, critic_tgt_path) = self.critic.save(path.join("critic"))?;
        let ent_coef_path = {
            let ent_coef_path = path.join("ent_coef.pt").to_path_buf();
            self.ent_coef.save(&ent_coef_path)?;
            ent_coef_path
        };

        Ok(vec![
            actor_path,
            critic_path,
            critic_tgt_path,
            ent_coef_path,
        ])
    }

    fn load_params(&mut self, path: &Path) -> Result<()> {
        self.actor.load(path.join("actor").as_path())?;
        self.critic.load(path.join("critic").as_path())?;
        self.ent_coef.load(path.join("ent_coef.pt").as_path())?;

        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
    }
}
