//! DQN agent implemented with tch-rs.
use super::{config::DQNConfig, explorer::DQNExplorer, model::DQNModel};
use crate::{
    model::{ModelBase, SubModel},
    util::{track, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Batch, Env, Policy, ReplayBufferBase,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{fs, marker::PhantomData, path::Path};
use tch::{no_grad, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
/// DQN agent implemented with tch-rs.
pub struct DQN<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    <R::Batch as Batch>::ObsBatch: Into<Q::Input>,
    <R::Batch as Batch>::ActBatch: Into<Tensor>,
{
    pub(in crate::dqn) soft_update_interval: usize,
    pub(in crate::dqn) soft_update_counter: usize,
    pub(in crate::dqn) n_updates_per_opt: usize,
    pub(in crate::dqn) min_transitions_warmup: usize,
    pub(in crate::dqn) batch_size: usize,
    pub(in crate::dqn) qnet: DQNModel<Q>,
    pub(in crate::dqn) qnet_tgt: DQNModel<Q>,
    pub(in crate::dqn) train: bool,
    pub(in crate::dqn) phantom: PhantomData<(E, R)>,
    pub(in crate::dqn) discount_factor: f64,
    pub(in crate::dqn) tau: f64,
    pub(in crate::dqn) explorer: DQNExplorer,
    pub(in crate::dqn) device: Device,
    pub(in crate::dqn) n_opts: usize,
    pub(in crate::dqn) double_dqn: bool,
    pub(in crate::dqn) _clip_reward: Option<f64>,
}

impl<E, Q, R> DQN<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    <R::Batch as Batch>::ObsBatch: Into<Q::Input>,
    <R::Batch as Batch>::ActBatch: Into<Tensor>,
{
    /// Constructs DQN agent.
    ///
    /// This is used with non-vectorized environments.
    pub fn build(config: DQNConfig<Q>, device: Device) -> Self {
        let qnet = DQNModel::build(config.model_config, device);
        let qnet_tgt = qnet.clone();

        DQN {
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
            phantom: PhantomData,
        }
    }

    fn update_critic(&mut self, buffer: &mut R) -> f32 {
        let batch = buffer.batch(self.batch_size).unwrap();
        let (obs, act, next_obs, reward, is_done, ixs, weight) = batch.unpack();
        let obs = obs.into();
        let act = act.into().to(self.device);
        let next_obs = next_obs.into();
        let reward = Tensor::of_slice(&reward[..]).to(self.device);
        let is_done = Tensor::of_slice(&is_done[..]).to(self.device);

        let pred = {
            let x = self.qnet.forward(&obs);
            x.gather(-1, &act, false).squeeze()
        };

        let tgt = no_grad(|| {
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
            reward + (1 - is_done) * self.discount_factor * q
        });

        let loss = if let Some(ws) = weight {
            let n = ws.len() as i64;
            let td_errs = (&pred - &tgt).abs();
            let loss = Tensor::of_slice(&ws[..]).to(self.device) * &td_errs;
            let loss = loss.smooth_l1_loss(
                &Tensor::zeros(&[n], tch::kind::FLOAT_CPU).to(self.device),
                tch::Reduction::Mean,
                1.0,
            );
            self.qnet.backward_step(&loss);
            let td_errs = Vec::<f32>::from(td_errs);
            buffer.update_priority(&ixs, &Some(td_errs));
            loss
        } else {
            let loss = pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0);
            self.qnet.backward_step(&loss);
            loss
        };

        f32::from(loss)
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
            track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
        }

        loss_critic /= self.n_updates_per_opt as f32;

        self.n_opts += 1;

        Record::from_slice(&[("loss_critic", RecordValue::Scalar(loss_critic))])
    }
}

impl<E, Q, R> Policy<E> for DQN<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    <R::Batch as Batch>::ObsBatch: Into<Q::Input>,
    <R::Batch as Batch>::ActBatch: Into<Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        no_grad(|| {
            let a = self.qnet.forward(&obs.clone().into());
            let a = if self.train {
                match &mut self.explorer {
                    DQNExplorer::Softmax(softmax) => softmax.action(&a),
                    DQNExplorer::EpsilonGreedy(egreedy) => egreedy.action(&a),
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

impl<E, Q, R> Agent<E, R> for DQN<E, Q, R>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<Q::Input>,
    E::Act: From<Q::Output>,
    Q::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    <R::Batch as Batch>::ObsBatch: Into<Q::Input>,
    <R::Batch as Batch>::ActBatch: Into<Tensor>,
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
