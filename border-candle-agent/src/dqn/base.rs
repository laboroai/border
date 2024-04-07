//! DQN agent implemented with tch-rs.
use super::{config::DqnConfig, explorer::DqnExplorer, model::DqnModel};
use crate::{
    model::SubModel1,
    util::{smooth_l1_loss, track, CriticLoss, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Env, Policy, ReplayBufferBase, StdBatchBase,
};
use candle_core::{shape::D, DType, Device, Tensor};
use candle_nn::loss::mse;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use std::convert::TryFrom;
use std::{fs, marker::PhantomData, path::Path};

#[allow(clippy::upper_case_acronyms, dead_code)]
/// DQN agent implemented with tch-rs.
pub struct Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input>,
    <R::Batch as StdBatchBase>::ActBatch: Into<Tensor>,
{
    pub(in crate::dqn) soft_update_interval: usize,
    pub(in crate::dqn) soft_update_counter: usize,
    pub(in crate::dqn) n_updates_per_opt: usize,
    pub(in crate::dqn) min_transitions_warmup: usize,
    pub(in crate::dqn) batch_size: usize,
    pub(in crate::dqn) qnet: DqnModel<Q>,
    pub(in crate::dqn) qnet_tgt: DqnModel<Q>,
    pub(in crate::dqn) train: bool,
    pub(in crate::dqn) phantom: PhantomData<(E, R)>,
    pub(in crate::dqn) discount_factor: f64,
    pub(in crate::dqn) tau: f64,
    pub(in crate::dqn) explorer: DqnExplorer,
    pub(in crate::dqn) device: Device,
    pub(in crate::dqn) n_opts: usize,
    pub(in crate::dqn) double_dqn: bool,
    pub(in crate::dqn) _clip_reward: Option<f64>,
    pub(in crate::dqn) clip_td_err: Option<(f64, f64)>,
    pub(in crate::dqn) critic_loss: CriticLoss,
    rng: SmallRng,
}

impl<E, Q, R> Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input>,
    <R::Batch as StdBatchBase>::ActBatch: Into<Tensor>,
{
    fn update_critic(&mut self, buffer: &mut R) -> f32 {
        let batch = buffer.batch(self.batch_size).unwrap();
        let (obs, act, next_obs, reward, is_done, _ixs, weight) = batch.unpack();
        let obs = obs.into();
        let act = act.into().to_device(&self.device).unwrap();
        let next_obs = next_obs.into();
        let reward = Tensor::from_slice(&reward[..], &[reward.len()], &self.device).unwrap();
        let is_not_done = {
            let is_not_done = is_done
                .into_iter()
                .map(|v| (1 - v) as f32)
                .collect::<Vec<_>>();
            Tensor::from_slice(&is_not_done[..], &[is_not_done.len()], &self.device).unwrap()
        };
        let pred = {
            let x = self.qnet.forward(&obs);
            x.gather(&act, D::Minus1)
                .unwrap()
                .squeeze(D::Minus1)
                .unwrap()
        };

        let tgt = {
            let q = if self.double_dqn {
                let x = self.qnet.forward(&next_obs);
                let y = x.argmax(D::Minus1).unwrap();
                let tgt = self.qnet_tgt.forward(&next_obs);
                tgt.gather(&y, D::Minus1).unwrap()
            } else {
                let x = self.qnet_tgt.forward(&next_obs);
                let y = x.argmax(D::Minus1).unwrap();
                x.gather(&y.unsqueeze(D::Minus1).unwrap(), D::Minus1)
                    .unwrap()
            };

            reward + is_not_done * self.discount_factor * q.squeeze(D::Minus1).unwrap()
        }
        .unwrap()
        .detach();

        let loss = if let Some(_ws) = weight {
            // Prioritized weighting loss, will be implemented later
            panic!();
            // let n = ws.len() as i64;
            // let td_errs = match self.clip_td_err {
            //     None => (&pred - &tgt).abs(),
            //     Some((min, max)) => (&pred - &tgt).abs().clip(min, max),
            // };
            // let loss = Tensor::of_slice(&ws[..]).to(self.device) * &td_errs;
            // let loss = loss.smooth_l1_loss(
            //     &Tensor::zeros(&[n], tch::kind::FLOAT_CPU).to(self.device),
            //     tch::Reduction::Mean,
            //     1.0,
            // );
            // self.qnet.backward_step(&loss);
            // let td_errs = Vec::<f32>::from(td_errs);
            // buffer.update_priority(&ixs, &Some(td_errs));
            // loss
        } else {
            match self.critic_loss {
                CriticLoss::Mse => mse(&pred, &tgt).unwrap(),
                CriticLoss::SmoothL1 => smooth_l1_loss(&pred, &tgt).unwrap(),
            }
        };

        // Backprop
        self.qnet.backward_step(&loss).unwrap();

        f32::from(loss.to_scalar::<f32>().unwrap())
    }

    fn opt_(&mut self, buffer: &mut R) -> Record {
        let mut loss_critic = 0f32;

        for _ in 0..self.n_updates_per_opt {
            let loss = self.update_critic(buffer);
            loss_critic += loss;
        }

        self.soft_update_counter += 1;
        if self.soft_update_counter == self.soft_update_interval {
            self.soft_update_counter = 0;
            let _ = track(self.qnet_tgt.get_varmap(), self.qnet.get_varmap(), self.tau);
        }

        loss_critic /= self.n_updates_per_opt as f32;

        self.n_opts += 1;

        Record::from_slice(&[("loss_critic", RecordValue::Scalar(loss_critic))])
    }
}

impl<E, Q, R> Policy<E> for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input>,
    <R::Batch as StdBatchBase>::ActBatch: Into<Tensor>,
{
    type Config = DqnConfig<Q>;

    /// Constructs DQN agent.
    fn build(config: Self::Config) -> Self {
        let device: Device = config
            .device
            .expect("No device is given for DQN agent")
            .into();
        let qnet = DqnModel::build(config.model_config.clone(), device.clone()).unwrap();
        let qnet_tgt = DqnModel::build(config.model_config, device.clone()).unwrap();

        Dqn {
            qnet,
            qnet_tgt,
            soft_update_interval: config.soft_update_interval,
            soft_update_counter: 0,
            n_updates_per_opt: config.n_updates_per_opt,
            min_transitions_warmup: config.min_transitions_warmup,
            batch_size: config.batch_size,
            discount_factor: config.discount_factor,
            tau: config.tau,
            train: config.train,
            explorer: config.explorer,
            device,
            n_opts: 0,
            _clip_reward: config.clip_reward,
            double_dqn: config.double_dqn,
            clip_td_err: config.clip_td_err,
            critic_loss: config.critic_loss,
            phantom: PhantomData,
            rng: SmallRng::seed_from_u64(42),
        }
    }

    /// In evaluation mode, take a random action with probability 0.01.
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let a = self.qnet.forward(&obs.clone().into());
        let a = if self.train {
            match &mut self.explorer {
                DqnExplorer::Softmax(softmax) => softmax.action(&a, &mut self.rng),
                DqnExplorer::EpsilonGreedy(egreedy) => egreedy.action(&a, &mut self.rng),
            }
        } else {
            if self.rng.gen::<f32>() < 0.01 {
                let n_actions = a.dims()[1] as i64;
                let a: i64 = self.rng.gen_range(0..n_actions);
                Tensor::try_from(vec![a]).unwrap()
            } else {
                a.argmax(D::Minus1).unwrap().to_dtype(DType::I64).unwrap()
            }
        };
        a.into()
    }
}

impl<E, Q, R> Agent<E, R> for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input>,
    <R::Batch as StdBatchBase>::ActBatch: Into<Tensor>,
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
        self.qnet.save(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt
            .save(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.qnet.load(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt
            .load(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }
}

#[cfg(feature = "border-async-trainer")]
use {crate::util::NamedTensors, border_async_trainer::SyncModel};

#[cfg(feature = "border-async-trainer")]
impl<E, Q, R> SyncModel for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: StdBatchBase,
    <R::Batch as StdBatchBase>::ObsBatch: Into<Q::Input>,
    <R::Batch as StdBatchBase>::ActBatch: Into<Tensor>,
{
    type ModelInfo = NamedTensors;

    fn model_info(&self) -> (usize, Self::ModelInfo) {
        (
            self.n_opts,
            NamedTensors::copy_from(self.qnet.get_var_store()),
        )
    }

    fn sync_model(&mut self, model_info: &Self::ModelInfo) {
        let vs = self.qnet.get_var_store_mut();
        model_info.copy_to(vs);
    }
}
