//! DQN agent implemented with candle.
use super::{config::DqnConfig, explorer::DqnExplorer, model::DqnModel};
use crate::{
    model::SubModel1,
    util::{smooth_l1_loss, track, CriticLoss, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Configurable, Env, Policy, ReplayBufferBase, TransitionBatch,
};
use candle_core::{shape::D, DType, Device, Tensor};
use candle_nn::loss::mse;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use std::convert::TryFrom;
use std::{fs, marker::PhantomData, path::Path};

#[allow(clippy::upper_case_acronyms, dead_code)]
/// DQN agent implemented with candle.
pub struct Dqn<E, Q, R>
where
    Q: SubModel1<Output = Tensor>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    pub(in crate::dqn) soft_update_interval: usize,
    pub(in crate::dqn) soft_update_counter: usize,
    pub(in crate::dqn) n_updates_per_opt: usize,
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
    n_samples_act: usize,
    n_samples_best_act: usize,
    record_verbose_level: usize,
    rng: SmallRng,
}

impl<E, Q, R> Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    fn update_critic(&mut self, buffer: &mut R) -> Record {
        let mut record = Record::empty();
        let batch = buffer.batch(self.batch_size).unwrap();
        let (obs, act, next_obs, reward, is_terminated, _is_truncated, _ixs, weight) =
            batch.unpack();
        let obs = obs.into();
        let act = act.into().to_device(&self.device).unwrap();
        let next_obs = next_obs.into();
        let reward = Tensor::from_slice(&reward[..], &[reward.len()], &self.device).unwrap();
        let is_not_terminated = {
            let is_not_terminated = is_terminated
                .into_iter()
                .map(|v| (1 - v) as f32)
                .collect::<Vec<_>>();
            Tensor::from_slice(
                &is_not_terminated[..],
                &[is_not_terminated.len()],
                &self.device,
            )
            .unwrap()
        };
        let pred = {
            let x = self.qnet.forward(&obs);
            x.gather(&act, D::Minus1)
                .unwrap()
                .squeeze(D::Minus1)
                .unwrap()
        };

        if self.record_verbose_level >= 2 {
            record.insert(
                "pred_mean",
                RecordValue::Scalar(pred.mean_all().unwrap().to_vec0::<f32>().unwrap()),
            );
        }

        if self.record_verbose_level >= 2 {
            let reward_mean: f32 = reward.mean_all().unwrap().to_vec0().unwrap();
            record.insert("reward_mean", RecordValue::Scalar(reward_mean));
        }

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

            reward + is_not_terminated * self.discount_factor * q.squeeze(D::Minus1).unwrap()
        }
        .unwrap()
        .detach();

        if self.record_verbose_level >= 2 {
            record.insert(
                "tgt_mean",
                RecordValue::Scalar(tgt.mean_all().unwrap().to_vec0::<f32>().unwrap()),
            );
            let tgt_minus_pred_mean: f32 = (&tgt - &pred)
                .unwrap()
                .mean_all()
                .unwrap()
                .to_vec0()
                .unwrap();
            record.insert(
                "tgt_minus_pred_mean",
                RecordValue::Scalar(tgt_minus_pred_mean),
            );
        }

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

        record.insert(
            "loss",
            RecordValue::Scalar(loss.to_scalar::<f32>().unwrap()),
        );

        record
        // f32::from(loss.to_scalar::<f32>().unwrap())
    }

    fn opt_(&mut self, buffer: &mut R) -> Record {
        let mut record_ = Record::empty();

        for _ in 0..self.n_updates_per_opt {
            let record = self.update_critic(buffer);
            record_ = record_.merge(record);
        }

        self.soft_update_counter += 1;
        if self.soft_update_counter == self.soft_update_interval {
            self.soft_update_counter = 0;
            let _ = track(self.qnet_tgt.get_varmap(), self.qnet.get_varmap(), self.tau);
        }

        self.n_opts += 1;

        record_
        // Record::from_slice(&[("loss", RecordValue::Scalar(loss_critic))])
    }
}

impl<E, Q, R> Policy<E> for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    /// In evaluation mode, take a random action with probability 0.01.
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let a = self.qnet.forward(&obs.clone().into()).detach();
        let a = if self.train {
            self.n_samples_act += 1;
            match &mut self.explorer {
                DqnExplorer::Softmax(softmax) => softmax.action(&a, &mut self.rng),
                DqnExplorer::EpsilonGreedy(egreedy) => {
                    if self.record_verbose_level >= 2 {
                        let (act, best) = egreedy.action_with_best(&a, &mut self.rng);
                        if best {
                            self.n_samples_best_act += 1;
                        }
                        act
                    } else {
                        egreedy.action(&a, &mut self.rng)
                    }
                }
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

impl<E, Q, R> Configurable for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel1<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    type Config = DqnConfig<Q>;

    /// Constructs DQN agent.
    fn build(config: Self::Config) -> Self {
        let device: Device = config
            .device
            .expect("No device is given for DQN agent")
            .into();
        let qnet = DqnModel::build(config.model_config.clone(), device.clone()).unwrap();
        let qnet_tgt = DqnModel::build(config.model_config.clone(), device.clone()).unwrap();
        let _ = track(qnet_tgt.get_varmap(), qnet.get_varmap(), 1.0);

        Dqn {
            qnet,
            qnet_tgt,
            soft_update_interval: config.soft_update_interval,
            soft_update_counter: 0,
            n_updates_per_opt: config.n_updates_per_opt,
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
            n_samples_act: 0,
            n_samples_best_act: 0,
            record_verbose_level: config.record_verbose_level,
            rng: SmallRng::seed_from_u64(42),
        }
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
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
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

    fn opt(&mut self, buffer: &mut R) {
        self.opt_(buffer);
    }

    fn opt_with_record(&mut self, buffer: &mut R) -> Record {
        let mut record = {
            let record = self.opt_(buffer);

            match self.record_verbose_level >= 2 {
                true => {
                    let record_weights = self.qnet.param_stats();
                    let record = record.merge(record_weights);
                    record
                }
                false => record,
            }
        };

        // Best action ratio for epsilon greedy
        let ratio = match self.n_samples_act == 0 {
            true => 0f32,
            false => self.n_samples_best_act as f32 / self.n_samples_act as f32,
        };
        record.insert("ratio_best_act", RecordValue::Scalar(ratio));
        self.n_samples_act = 0;
        self.n_samples_best_act = 0;

        record
    }

    /// Save model parameters in the given directory.
    ///
    /// The parameters of the model are saved as `qnet.pt`.
    /// The parameters of the target model are saved as `qnet_tgt.pt`.
    fn save_params<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        self.qnet.save(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt
            .save(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }

    fn load_params<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
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
    Q: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    type ModelInfo = NamedTensors;

    fn model_info(&self) -> (usize, Self::ModelInfo) {
        unimplemented!();
        // (
        //     self.n_opts,
        //     NamedTensors::copy_from(self.qnet.get_var_store()),
        // )
    }

    fn sync_model(&mut self, _model_info: &Self::ModelInfo) {
        unimplemented!();
        // let vs = self.qnet.get_var_store_mut();
        // model_info.copy_to(vs);
    }
}
