//! DQN agent implemented with tch-rs.
use super::{config::DqnConfig, explorer::DqnExplorer, model::DqnModel};
use crate::{
    model::{ModelBase, SubModel},
    util::{track, CriticLoss, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Configurable, Env, Policy, ReplayBufferBase, TransitionBatch,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    convert::{TryFrom, TryInto},
    fs,
    marker::PhantomData,
    path::Path,
};
use tch::{no_grad, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
/// DQN agent implemented with tch-rs.
pub struct Dqn<E, Q, R>
where
    Q: SubModel<Output = Tensor>,
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
}

impl<E, Q, R> Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    R: ReplayBufferBase,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    fn update_critic(&mut self, buffer: &mut R) -> Record {
        let mut record = Record::empty();
        let batch = buffer.batch(self.batch_size).unwrap();
        let (obs, act, next_obs, reward, is_terminated, _is_truncated, ixs, weight) =
            batch.unpack();
        let obs = obs.into();
        let act = act.into().to(self.device);
        let next_obs = next_obs.into();
        let reward = Tensor::from_slice(&reward[..]).to(self.device);
        let is_terminated = Tensor::from_slice(&is_terminated[..]).to(self.device);

        let pred = {
            let x = self.qnet.forward(&obs);
            x.gather(-1, &act, false).squeeze()
        };

        if self.record_verbose_level >= 2 {
            record.insert(
                "pred_mean",
                RecordValue::Scalar(
                    f32::try_from(pred.mean(tch::Kind::Float))
                        .expect("Failed to convert Tensor to f32"),
                ),
            );
        }

        if self.record_verbose_level >= 2 {
            let reward_mean: f32 = reward.mean(tch::Kind::Float).try_into().unwrap();
            record.insert("reward_mean", RecordValue::Scalar(reward_mean));
        }

        let tgt: Tensor = no_grad(|| {
            let q = if self.double_dqn {
                let x = self.qnet.forward(&next_obs);
                let y = x.argmax(-1, false).unsqueeze(-1);
                self.qnet_tgt
                    .forward(&next_obs)
                    .gather(-1, &y, false)
                    .squeeze()
            } else {
                let x = self.qnet_tgt.forward(&next_obs);
                let y = x.argmax(-1, false).unsqueeze(-1);
                x.gather(-1, &y, false).squeeze()
            };
            reward + (1 - is_terminated) * self.discount_factor * q
        });

        if self.record_verbose_level >= 2 {
            record.insert(
                "tgt_mean",
                RecordValue::Scalar(
                    f32::try_from(tgt.mean(tch::Kind::Float))
                        .expect("Failed to convert Tensor to f32"),
                ),
            );
            let tgt_minus_pred_mean: f32 =
                (&tgt - &pred).mean(tch::Kind::Float).try_into().unwrap();
            record.insert(
                "tgt_minus_pred_mean",
                RecordValue::Scalar(tgt_minus_pred_mean),
            );
        }

        let loss = if let Some(ws) = weight {
            let n = ws.len() as i64;
            let td_errs = match self.clip_td_err {
                None => (&pred - &tgt).abs(),
                Some((min, max)) => (&pred - &tgt).abs().clip(min, max),
            };
            let loss = Tensor::from_slice(&ws[..]).to(self.device) * &td_errs;
            let loss = match self.critic_loss {
                CriticLoss::SmoothL1 => loss.smooth_l1_loss(
                    &Tensor::zeros(&[n], tch::kind::FLOAT_CPU).to(self.device),
                    tch::Reduction::Mean,
                    1.0,
                ),
                CriticLoss::Mse => loss.mse_loss(
                    &Tensor::zeros(&[n], tch::kind::FLOAT_CPU).to(self.device),
                    tch::Reduction::Mean,
                ),
            };
            self.qnet.backward_step(&loss);
            let td_errs = Vec::<f32>::try_from(td_errs).expect("Failed to convert Tensor to f32");
            buffer.update_priority(&ixs, &Some(td_errs));
            loss
        } else {
            let loss = match self.critic_loss {
                CriticLoss::SmoothL1 => pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0),
                CriticLoss::Mse => pred.mse_loss(&tgt, tch::Reduction::Mean),
            };
            self.qnet.backward_step(&loss);
            loss
        };

        record.insert(
            "loss",
            RecordValue::Scalar(f32::try_from(loss).expect("Failed to convert Tensor to f32")),
        );

        record
    }

    // fn opt_(&mut self, buffer: &mut R) -> Record {
    //     let mut loss = 0f32;

    //     for _ in 0..self.n_updates_per_opt {
    //         loss += self.update_critic(buffer);
    //     }

    //     self.soft_update_counter += 1;
    //     if self.soft_update_counter == self.soft_update_interval {
    //         self.soft_update_counter = 0;
    //         track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
    //     }

    //     loss /= self.n_updates_per_opt as f32;

    //     self.n_opts += 1;

    //     Record::from_slice(&[("loss", RecordValue::Scalar(loss))])
    // }

    fn opt_(&mut self, buffer: &mut R) -> Record {
        let mut record_ = Record::empty();

        for _ in 0..self.n_updates_per_opt {
            let record = self.update_critic(buffer);
            record_ = record_.merge(record);
        }

        self.soft_update_counter += 1;
        if self.soft_update_counter == self.soft_update_interval {
            self.soft_update_counter = 0;
            track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
        }

        self.n_opts += 1;

        record_
        // Record::from_slice(&[("loss", RecordValue::Scalar(loss_critic))])
    }
}

impl<E, Q, R> Policy<E> for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        no_grad(|| {
            let a = self.qnet.forward(&obs.clone().into());
            let a = if self.train {
                self.n_samples_act += 1;
                match &mut self.explorer {
                    DqnExplorer::Softmax(softmax) => softmax.action(&a),
                    DqnExplorer::EpsilonGreedy(egreedy) => {
                        if self.record_verbose_level >= 2 {
                            let (act, best) = egreedy.action_with_best(&a);
                            if best {
                                self.n_samples_best_act += 1;
                            }
                            act
                        } else {
                            egreedy.action(&a)
                        }
                    }
                }
            } else {
                if fastrand::f32() < 0.01 {
                    let n_actions = a.size()[1] as i32;
                    let a = fastrand::i32(0..n_actions);
                    Tensor::from(a)
                } else {
                    a.argmax(-1, true)
                }
            };
            a.into()
        })
    }
}

impl<E, Q, R> Configurable for Dqn<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    type Config = DqnConfig<Q>;

    /// Constructs DQN agent.
    fn build(config: Self::Config) -> Self {
        let device = config
            .device
            .expect("No device is given for DQN agent")
            .into();
        let qnet = DqnModel::build(config.model_config, device);
        let qnet_tgt = qnet.clone();

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
            n_samples_act: 0,
            n_samples_best_act: 0,
            record_verbose_level: config.record_verbose_level,
            phantom: PhantomData,
        }
    }
}

impl<E, Q, R> Agent<E, R> for Dqn<E, Q, R>
where
    E: Env + 'static,
    Q: SubModel<Output = Tensor> + 'static,
    R: ReplayBufferBase + 'static,
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
        if self.record_verbose_level >= 2 {
            let ratio = match self.n_samples_act == 0 {
                true => 0f32,
                false => self.n_samples_best_act as f32 / self.n_samples_act as f32,
            };
            record.insert("ratio_best_act", RecordValue::Scalar(ratio));
            self.n_samples_act = 0;
            self.n_samples_best_act = 0;
        }

        record
    }

    /// Save model parameters in the given directory.
    ///
    /// The parameters of the model are saved as `qnet.pt`.
    /// The parameters of the target model are saved as `qnet_tgt.pt`.
    fn save_params(&self, path: &Path) -> Result<()> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        self.qnet.save(path.join("qnet.pt.tch").as_path())?;
        self.qnet_tgt.save(path.join("qnet_tgt.pt.tch").as_path())?;
        Ok(())
    }

    fn load_params(&mut self, path: &Path) -> Result<()> {
        self.qnet.load(path.join("qnet.pt.tch").as_path())?;
        self.qnet_tgt.load(path.join("qnet_tgt.pt.tch").as_path())?;
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
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
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<Q::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
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
