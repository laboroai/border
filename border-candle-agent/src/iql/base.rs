use super::{IqlConfig, Value};
use crate::{
    model::{SubModel1, SubModel2},
    util::{
        actor::GaussianActor, asymmetric_l2_loss, critic::MultiCritic, gamma_not_done, reward,
        smooth_l1_loss, CriticLoss, OutDim,
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
type StateValue = Tensor; // expectile
type ActMean = Tensor;
type ActStd = Tensor;

/// Implicit Q-learning (IQL) agent.
pub struct Iql<E, Q, P, V, R, O, A>
where
    Q: SubModel2<Input1 = O, Input2 = A, Output = ActionValue>,
    P: SubModel1<Input = O, Output = (ActMean, ActStd)>,
    V: SubModel1<Input = O, Output = StateValue>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    critic: MultiCritic<Q>,
    actor: GaussianActor<P>,
    value: Value<V>,
    gamma: f32,
    tau_iql: f64, // expectile percent
    inv_lambda: f64,
    n_updates_per_opt: usize,
    batch_size: usize,
    train: bool,
    // reward_scale: f32,
    exp_adv_max: f64,
    critic_loss: CriticLoss,
    device: Device,
    adv_softmax: bool,
    n_opts: usize,
    phantom: PhantomData<(E, R, O, A)>,
}

impl<E, Q, P, V, R, O, A> Iql<E, Q, P, V, R, O, A>
where
    E: Env,
    Q: SubModel2<Input1 = O, Input2 = A, Output = ActionValue>,
    P: SubModel1<Input = O, Output = (ActMean, ActStd)>,
    V: SubModel1<Input = O, Output = StateValue>,
    R: ReplayBufferBase,
    E::Obs: Into<O>,
    E::Act: Into<A>,
    A: Clone,
    Q::Input2: From<ActMean> + Into<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch:
        Into<Q::Input1> + Into<P::Input> + Into<V::Input> + Clone,
    <R::Batch as TransitionBatch>::ActBatch: Into<Q::Input2> + Into<Tensor> + Clone,
{
    fn update_value(&mut self, obs: &O, act: &A) -> Result<f32> {
        let loss = {
            let q = self.critic.qvals_min_tgt(obs, act)?.detach();
            let v = self.value.forward(obs).squeeze(D::Minus1);
            let u = (q - v)?;
            asymmetric_l2_loss(&u, self.tau_iql)?
        };

        self.value.backward_step(&loss)?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn update_critic(
        &mut self,
        obs: &O,
        act: &A,
        next_obs: &O,
        gamma_not_done: &Tensor,
        reward: &Tensor,
    ) -> Result<f32> {
        let loss = {
            // Prediction
            let preds = self.critic.qvals(obs, act);

            // Target
            let tgt = (reward
                + (gamma_not_done * self.value.forward(next_obs).squeeze(D::Minus1)?)?)?
            .detach();
            debug_assert_eq!(tgt.dims(), [self.batch_size]);

            // Loss
            let losses: Vec<_> = match self.critic_loss {
                CriticLoss::Mse => preds.iter().map(|pred| mse(&pred, &tgt).unwrap()).collect(),
                CriticLoss::SmoothL1 => preds
                    .iter()
                    .map(|pred| smooth_l1_loss(&pred, &tgt).unwrap())
                    .collect(),
            };
            Tensor::stack(&losses, 0)?.mean_all()?
        };

        self.critic.backward_step(&loss)?;
        self.critic.soft_update()?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn update_actor(&mut self, obs: &O, act: &A) -> Result<f32> {
        let loss = {
            // Weights
            let w = {
                let q = self.critic.qvals_min_tgt(&obs, &act)?;
                let v = self.value.forward(&obs).squeeze(D::Minus1)?;
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

            // Log probability of actions in the batch
            log::trace!("Log probability of actions in the batch");
            let logp = self.actor.logp(obs, &(*act).clone().into())?;
            debug_assert_eq!(logp.dims(), &[self.batch_size]);

            // Loss
            log::trace!("Loss");
            (-1f64 * logp * w)?.mean_all()?
        };

        self.actor.backward_step(&loss)?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn opt_(&mut self, buffer: &mut R) -> Result<Record> {
        let mut loss_value = 0f32;
        let mut loss_critic = 0f32;
        let mut loss_actor = 0f32;

        for _ in 0..self.n_updates_per_opt {
            let batch = buffer.batch(self.batch_size).unwrap();
            let (obs, act, next_obs, _reward, is_terminated, is_truncated, _, _) = batch.unpack();
            let reward = reward(_reward, &self.device)?;
            let gnd = gamma_not_done(self.gamma, is_terminated, is_truncated, &self.device)?;
            let obs = &obs.into();
            let act = &act.into();
            let next_obs = &next_obs.into();

            loss_value += self.update_value(obs, act)?;
            loss_critic += self.update_critic(obs, act, next_obs, &gnd, &reward)?;
            loss_actor += self.update_actor(obs, act)?;
            self.n_opts += 1;
        }

        loss_value /= self.n_updates_per_opt as f32;
        loss_critic /= self.n_updates_per_opt as f32;
        loss_actor /= self.n_updates_per_opt as f32;

        let record = Record::from_slice(&[
            ("loss_value", RecordValue::Scalar(loss_value)),
            ("loss_critic", RecordValue::Scalar(loss_critic)),
            ("loss_actor", RecordValue::Scalar(loss_actor)),
        ]);

        Ok(record)
    }
}

impl<E, Q, P, V, R, O, A> Policy<E> for Iql<E, Q, P, V, R, O, A>
where
    E: Env,
    Q: SubModel2<Input1 = O, Input2 = A, Output = ActionValue>,
    P: SubModel1<Input = O, Output = (ActMean, ActStd)>,
    V: SubModel1<Input = O, Output = StateValue>,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        self.actor
            .sample(&obs.clone().into(), self.train)
            .unwrap()
            .into()
    }
}

impl<E, Q, P, V, R, O, A> Configurable for Iql<E, Q, P, V, R, O, A>
where
    E: Env,
    Q: SubModel2<Input1 = O, Input2 = A, Output = ActionValue>,
    P: SubModel1<Input = O, Output = (ActMean, ActStd)>,
    V: SubModel1<Input = O, Output = StateValue>,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    type Config = IqlConfig<Q, P, V>;

    /// Constructs [`Iql`] agent.
    fn build(config: Self::Config) -> Self {
        let device: Device = config
            .device
            .expect("No device is given for IQL agent")
            .into();
        let critic = MultiCritic::build(config.critic_config, device.clone()).unwrap();
        let actor = GaussianActor::build(config.actor_config, device.clone().into()).unwrap();
        let value = Value::build(config.value_config, device.clone()).unwrap();

        Iql {
            critic,
            actor,
            value,
            gamma: config.gamma,
            tau_iql: config.tau_iql,
            inv_lambda: config.inv_lambda,
            n_updates_per_opt: config.n_updates_per_opt,
            batch_size: config.batch_size,
            // reward_scale: config.reward_scale,
            critic_loss: config.critic_loss,
            exp_adv_max: config.exp_adv_max,
            n_opts: 0,
            train: false,
            device: device.into(),
            adv_softmax: config.adv_softmax,
            phantom: PhantomData,
        }
    }
}

impl<E, Q, P, V, R, O, A> Agent<E, R> for Iql<E, Q, P, V, R, O, A>
where
    E: Env + 'static,
    Q: SubModel2<Input1 = O, Input2 = A, Output = ActionValue> + 'static,
    P: SubModel1<Input = O, Output = (ActMean, ActStd)> + 'static,
    V: SubModel1<Input = O, Output = StateValue> + 'static,
    R: ReplayBufferBase + 'static,
    E::Obs: Into<Q::Input1> + Into<P::Input>,
    E::Act: Into<Q::Input2> + From<Tensor>,
    O: 'static,
    A: Clone + 'static,
    Q::Input2: From<ActMean> + Into<Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    V::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch:
        Into<Q::Input1> + Into<P::Input> + Into<V::Input> + Clone,
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
        self.opt_(buffer).expect("Failed in Iql::opt_()")
    }

    fn save_params(&self, path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;

        let actor_path = self.actor.save(path.join("actor"))?;
        let (critic_path, critic_tgt_path) = self.critic.save(path.join("critic"))?;
        let value_path = self.value.save(path.join("value"))?;

        Ok(vec![actor_path, critic_path, critic_tgt_path, value_path])
    }

    fn load_params(&mut self, path: &Path) -> Result<()> {
        self.actor.load(path.join("actor").as_path())?;
        self.critic.load(path.join("critic").as_path())?;
        self.value.load(path.join("value").as_path())?;

        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
    }
}
