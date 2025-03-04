use super::AwacConfig;
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
use candle_nn::{loss::mse, ops::softmax};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs,
    marker::PhantomData,
    path::{Path, PathBuf},
};

type ActionValue = Tensor;
type ActMean = Tensor;
type ActStd = Tensor;

/// Advantage weighted actor critic (AWAC) agent.
pub struct Awac<E, Q, P, R>
where
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    critic: MultiCritic<Q>,
    actor: GaussianActor<P>,
    gamma: f64,
    inv_lambda: f64,
    n_updates_per_opt: usize,
    batch_size: usize,
    train: bool,
    // reward_scale: f32,
    n_opts: usize,
    exp_adv_max: f64,
    critic_loss: CriticLoss,
    phantom: PhantomData<(E, R)>,
    device: Device,
    adv_softmax: bool,
}

impl<E, Q, P, R> Awac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + Into<Tensor>,
    Q::Input2: From<ActMean> + Into<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as TransitionBatch>::ActBatch: Into<Q::Input2> + Into<Tensor> + Clone,
{
    fn update_critic(&mut self, batch: R::Batch) -> Result<(f32, f32)> {
        let (loss, q_tgt_abs_mean) = {
            // Extract items in the batch
            let (obs, act, next_obs, reward, is_terminated, is_truncated, _, _) = batch.unpack();
            let batch_size = reward.len();
            let reward = Tensor::from_slice(&reward[..], (batch_size,), &self.device)?;

            // Prediction
            let qs = self.critic.qvals(&obs.into(), &act.into());

            // Target
            let tgt = {
                let gamma_not_done = gamma_not_done(
                    self.gamma as f32,
                    is_terminated,
                    Some(is_truncated),
                    &self.device,
                )?;
                let next_act = self.actor.sample(&next_obs.clone().into(), self.train)?;
                let next_q = self
                    .critic
                    .qvals_min_tgt(&next_obs.into(), &next_act.into())?;
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

            // for debug
            let q_tgt_abs_mean = tgt.abs()?.mean_all()?;

            (Tensor::stack(&losses, 0)?.sum_all()?, q_tgt_abs_mean)
        };

        self.critic.backward_step(&loss)?;
        self.critic.soft_update()?;

        Ok((loss.to_scalar::<f32>()?, q_tgt_abs_mean.to_scalar::<f32>()?))
    }

    fn update_actor(&mut self, batch: &R::Batch) -> Result<f32> {
        // Extract items in the batch
        log::trace!("Extract items in the batch");
        let obs = batch.obs().clone();
        let act = batch.act().clone();

        let w = {
            let act_ = self.actor.sample(&obs.clone().into(), self.train)?;
            let q = self
                .critic
                .qvals_min_tgt(&obs.clone().into(), &act.clone().into())?;
            let v = self
                .critic
                .qvals_min_tgt(&obs.clone().into(), &act_.into())?;
            let adv = (&q - &v)?;
            debug_assert_eq!(adv.dims(), &[self.batch_size]);

            match self.adv_softmax {
                false => (adv * self.inv_lambda)?
                    .exp()?
                    .clamp(0f64, self.exp_adv_max)?,
                true => softmax(&(adv * self.inv_lambda)?, 0)?,
            }
        }
        .detach();
        debug_assert_eq!(w.dims(), &[self.batch_size]);

        let loss = {
            let logp = self.actor.logp(&obs.into(), &act.into())?;
            debug_assert_eq!(logp.dims(), &[self.batch_size]);

            (-1f64 * logp * w)?.mean_all()?
        };

        self.actor.backward_step(&loss)?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn opt_(&mut self, buffer: &mut R) -> Result<Record> {
        let mut loss_critic = 0f32;
        let mut loss_actor = 0f32;
        let mut q_tgt_abs_mean = 0f32;

        for _ in 0..self.n_updates_per_opt {
            let batch = buffer.batch(self.batch_size).unwrap();
            loss_actor += self.update_actor(&batch)?;

            // loss_critic += self.update_critic(batch)?;
            let (loss_critic_, q_tgt_abs_mean_) = self.update_critic(batch)?;
            loss_critic += loss_critic_;
            q_tgt_abs_mean += q_tgt_abs_mean_;
            self.n_opts += 1;
        }

        loss_critic /= self.n_updates_per_opt as f32;
        loss_actor /= self.n_updates_per_opt as f32;

        let record = Record::from_slice(&[
            ("loss_critic", RecordValue::Scalar(loss_critic)),
            ("loss_actor", RecordValue::Scalar(loss_actor)),
            ("q_tgt_abs_mean", RecordValue::Scalar(q_tgt_abs_mean)),
        ]);

        Ok(record)
    }
}

impl<E, Q, P, R> Policy<E> for Awac<E, Q, P, R>
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

impl<E, Q, P, R> Configurable for Awac<E, Q, P, R>
where
    E: Env,
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    type Config = AwacConfig<Q, P>;

    /// Constructs [`Awac`] agent.
    fn build(config: Self::Config) -> Self {
        let device: Device = config
            .device
            .expect("No device is given for AWAC agent")
            .into();
        let actor = GaussianActor::build(config.actor_config, device.clone().into()).unwrap();
        let critics = MultiCritic::build(config.critic_config, device.clone().into()).unwrap();

        Awac {
            critic: critics,
            actor,
            gamma: config.gamma,
            // action_min: config.action_min,
            // action_max: config.action_max,
            // min_lstd: config.min_lstd,
            // max_lstd: config.max_lstd,
            n_updates_per_opt: config.n_updates_per_opt,
            batch_size: config.batch_size,
            // reward_scale: config.reward_scale,
            critic_loss: config.critic_loss,
            inv_lambda: config.inv_lambda,
            exp_adv_max: config.exp_adv_max,
            n_opts: 0,
            train: false,
            device: device.into(),
            adv_softmax: config.adv_softmax,
            phantom: PhantomData,
        }
    }
}

impl<E, Q, P, R> Agent<E, R> for Awac<E, Q, P, R>
where
    E: Env + 'static,
    Q: SubModel2<Output = ActionValue> + 'static,
    P: SubModel1<Output = (ActMean, ActStd)> + 'static,
    R: ReplayBufferBase + 'static,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + Into<Tensor> + From<Tensor>,
    Q::Input2: From<ActMean> + Into<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input1> + Into<P::Input> + Clone,
    <R::Batch as TransitionBatch>::ActBatch: Into<Q::Input2> + Into<Tensor> + Clone,
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
        self.opt_(buffer).expect("Failed in Awac::opt_()")
    }

    fn save_params(&self, path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;

        let actor_path = self.actor.save(path.join("actor"))?;
        let (critic_path, critic_tgt_path) = self.critic.save(path.join("critic"))?;

        Ok(vec![actor_path, critic_path, critic_tgt_path])
    }

    fn load_params(&mut self, path: &Path) -> Result<()> {
        self.actor.load(path.join("actor").as_path())?;
        self.critic.load(path.join("critic").as_path())?;

        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
    }
}
