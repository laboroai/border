use super::{Actor, AwacConfig, Critic};
use crate::{
    model::{SubModel1, SubModel2},
    util::{smooth_l1_loss, track, CriticLoss, OutDim},
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

fn normal_logp(x: &Tensor) -> Result<Tensor> {
    let tmp: Tensor =
        ((-0.5 * (2.0 * std::f32::consts::PI).ln() as f64) - (0.5 * x.powf(2.0)?)?)?;
    Ok(tmp.sum(D::Minus1)?)
}

/// Advantage weighted actor critic (AWAC) agent.
pub struct Awac<E, Q, P, R>
where
    Q: SubModel2<Output = ActionValue>,
    P: SubModel1<Output = (ActMean, ActStd)>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    critics: Vec<Critic<Q>>,
    critics_tgt: Vec<Critic<Q>>,
    actor: Actor<P>,
    gamma: f64,
    tau: f64,
    inv_lambda: f64,
    action_min: f32,
    action_max: f32,
    min_lstd: f64,
    max_lstd: f64,
    n_updates_per_opt: usize,
    batch_size: usize,
    train: bool,
    reward_scale: f32,
    n_opts: usize,
    exp_adv_max: f64,
    critic_loss: CriticLoss,
    phantom: PhantomData<(E, R)>,
    device: Device,
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
    #[allow(dead_code)]
    /// ArcTanh
    fn atanh(t: &Tensor) -> Result<Tensor> {
        let t = t.clamp(-0.999999, 0.999999)?;
        Ok((0.5 * (((1. + &t)? / (1. - &t)?)?).log()?)?)
    }

    #[allow(dead_code)]
    /// Density transformation for tanh function
    fn log_jacobian(&self, a: &Tensor, eps: f64) -> Result<Tensor> {
        let eps = Tensor::new(&[eps as f32], &self.device)?.broadcast_as(a.shape())?;
        // let eps = Tensor::new(&[eps], self.device)?.broadcast_as(a.shape())?;
        Ok((-1f64 * (1f64 - a.powf(2.0)? + eps)?.log()?)?.sum(D::Minus1)?)
    }

    /// Rerurns the log probabilities (densities) of the given actions
    fn logp<'a>(&self, obs: &P::Input, act: Q::Input2) -> Result<Tensor> {
        // Distribution parameters on the given observation
        log::trace!("Distribution parameters on the given observation");
        let (mean, std) = {
            let (mean, lstd) = self.actor.forward(obs);
            let std = lstd.clamp(self.min_lstd, self.max_lstd)?.exp()?;
            (mean, std)
        };

        // Inverse transformation to the standard normal: N(0, 1)
        log::trace!("Inverse transformation to the standard normal: N(0, 1)");
        let act = act.into().to_device(&self.device)?;
        // let z = ((Self::atanh(&act)? - &mean)? / &std)?;
        let z = ((&act - &mean)? / &std)?;

        // Density
        log::trace!("Density");
        // Ok((normal_logp(&z)? + self.log_jacobian(&act, self.epsilon)?)?) // tanh trans.
        Ok(normal_logp(&z)?)
    }

    /// Returns an action and its log probability based on the Normal distribution.
    fn action_logp(&self, o: &P::Input) -> Result<(Tensor, Tensor)> {
        let (mean, lstd) = self.actor.forward(o);
        let std = lstd.clamp(self.min_lstd, self.max_lstd)?.exp()?;
        let z = Tensor::randn(0f32, 1f32, mean.dims(), &self.device)?;
        let a = (&std * &z + &mean)?.clamp(self.action_min, self.action_max)?;
        let log_p = normal_logp(&z)?; // .sum(D::Minus1)?

        debug_assert_eq!(a.dims()[0], self.batch_size);
        debug_assert_eq!(log_p.dims(), [self.batch_size]);

        Ok((a, log_p))
    }

    fn qvals(&self, critics: &[Critic<Q>], obs: &Q::Input1, act: &Q::Input2) -> Vec<Tensor> {
        critics
            .iter()
            .map(|critic| critic.forward(obs, act).squeeze(D::Minus1).unwrap())
            .collect()
    }

    /// Returns the minimum values of q values over critics
    fn qvals_min(&self, qnets: &[Critic<Q>], obs: &Q::Input1, act: &Q::Input2) -> Result<Tensor> {
        let qvals = self.qvals(qnets, obs, act);
        let qvals = Tensor::stack(&qvals, D::Minus1)?;
        let qvals_min = qvals.min(D::Minus1)?.squeeze(D::Minus1)?;

        debug_assert_eq!(qvals_min.dims(), [self.batch_size]);

        Ok(qvals_min)
    }

    fn update_critic(&mut self, batch: R::Batch) -> Result<f32> {
        let losses = {
            // Extract items in the batch
            let (obs, act, next_obs, reward, is_terminated, _is_truncated, _, _) = batch.unpack();
            let batch_size = reward.len();
            let reward = Tensor::from_slice(&reward[..], (batch_size,), &self.device)?;
            let is_terminated = {
                let is_terminated = is_terminated.iter().map(|e| *e as f32).collect::<Vec<_>>();
                Tensor::from_slice(&is_terminated[..], (batch_size,), &self.device)?
            };

            // Prediction
            let qs = self.qvals(&self.critics, &obs.into(), &act.into());

            // Target
            let (a_next, _) = self.action_logp(&next_obs.clone().into())?;
            let a_next = a_next.detach();
            let q_next = self
                .qvals_min(&self.critics_tgt, &next_obs.into(), &a_next.into())?
                .detach();
            let tgt = (((self.reward_scale as f64) * reward)?
                + (1f64 - &is_terminated)? * self.gamma * q_next)?;

            debug_assert_eq!(tgt.dims(), [self.batch_size]);

            // Loss
            let losses: Vec<_> = match self.critic_loss {
                // CriticLoss::Mse => qs
                //     .iter()
                //     .map(|q| mse(&q.squeeze(D::Minus1).unwrap(), &tgt).unwrap())
                //     .collect(),
                CriticLoss::Mse => qs.iter().map(|pred| mse(&pred, &tgt).unwrap()).collect(),
                CriticLoss::SmoothL1 => qs
                    .iter()
                    .map(|pred| smooth_l1_loss(&pred, &tgt).unwrap())
                    .collect(),
            };
            losses
        };

        for (critic, loss) in self.critics.iter_mut().zip(&losses) {
            critic.backward_step(&loss).unwrap();
        }

        Ok(losses
            .iter()
            .map(|loss| loss.to_scalar::<f32>().unwrap())
            .sum::<f32>()
            / (self.critics.len() as f32))
    }

    fn update_actor(&mut self, batch: &R::Batch) -> Result<f32> {
        // Extract items in the batch
        log::trace!("Extract items in the batch");
        let obs = batch.obs().clone();
        let act = batch.act().clone();

        // Weights
        let w = {
            // Advantage
            let adv = {
                // Action from the current policy
                log::trace!("Action from the current policy");
                let (act_curr, _) = self.action_logp(&obs.clone().into())?;

                log::trace!("Advantage");
                let q = self.qvals_min(&self.critics, &obs.clone().into(), &act.clone().into())?;
                let v = self.qvals_min(&self.critics, &obs.clone().into(), &act_curr.into())?;
                (q - v)?
            };

            log::trace!("Weights");
            (adv * self.inv_lambda)?
                .exp()?
                .clamp(0f64, self.exp_adv_max)?
        }
        .detach();

        // Log probability of actions in the batch
        log::trace!("Log probability of actions in the batch");
        let logp = self.logp(&obs.into(), act.into())?;

        // Loss
        log::trace!("Loss");
        let loss = (-1f64 * logp * w)?.mean_all()?;

        self.actor.backward_step(&loss)?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn soft_update(&mut self) -> Result<()> {
        for (critic_tgt, critic) in self.critics_tgt.iter().zip(&mut self.critics) {
            track(critic_tgt.get_varmap(), critic.get_varmap(), self.tau)?;
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
        let obs = obs.clone().into();
        let (mean, lstd) = self.actor.forward(&obs);
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
        // act.tanh().unwrap().into()
        act.clamp(-1f32, 1f32).unwrap().into()
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
        let device = config.device.expect("No device is given for SAC agent");
        let n_critics = config.n_critics;
        let actor = Actor::build(config.actor_config, device.clone().into()).unwrap();
        let mut critics = vec![];
        let mut critics_tgt = vec![];
        for _ in 0..n_critics {
            critics
                .push(Critic::build(config.critic_config.clone(), device.clone().into()).unwrap());
            critics_tgt
                .push(Critic::build(config.critic_config.clone(), device.clone().into()).unwrap());
        }

        Awac {
            critics,
            critics_tgt,
            actor,
            gamma: config.gamma,
            tau: config.tau,
            action_min: config.action_min,
            action_max: config.action_max,
            min_lstd: config.min_lstd,
            max_lstd: config.max_lstd,
            n_updates_per_opt: config.n_updates_per_opt,
            batch_size: config.batch_size,
            reward_scale: config.reward_scale,
            critic_loss: config.critic_loss,
            inv_lambda: config.inv_lambda,
            exp_adv_max: config.exp_adv_max,
            n_opts: 0,
            train: false,
            device: device.into(),
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
        let mut paths = vec![];

        for (i, (critic, critic_tgt)) in self.critics.iter().zip(&self.critics_tgt).enumerate() {
            let path1 = path.join(format!("critic_{}.pt", i)).to_path_buf();
            let path2 = path.join(format!("critic_tgt_{}.pt", i)).to_path_buf();
            critic.save(&path1)?;
            critic_tgt.save(&path2)?;
            paths.push(path1);
            paths.push(path2);
        }
        let path_actor = path.join("actor.pt").to_path_buf();
        self.actor.save(&path_actor)?;
        paths.push(path_actor);

        Ok(paths)
    }

    fn load_params(&mut self, path: &Path) -> Result<()> {
        for (i, (critic, critic_tgt)) in self
            .critics
            .iter_mut()
            .zip(&mut self.critics_tgt)
            .enumerate()
        {
            critic.load(path.join(format!("critic_{}.pt", i)).as_path())?;
            critic_tgt.load(path.join(format!("critic_tgt_{}.pt", i)).as_path())?;
        }
        self.actor.load(path.join("actor.pt").as_path())?;
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
    }
}
