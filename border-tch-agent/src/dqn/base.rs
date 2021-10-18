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
    // pub(in crate::dqn) expr_sampling: ExperienceSampling,
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
            // replay_buffer,
            // prev_obs: RefCell::new(None),
            // opt_interval_counter: self.opt_interval_counter,
            soft_update_interval: config.soft_update_interval,
            soft_update_counter: 0,
            n_updates_per_opt: config.n_updates_per_opt,
            min_transitions_warmup: config.min_transitions_warmup,
            batch_size: config.batch_size,
            discount_factor: config.discount_factor,
            tau: config.tau,
            train: config.train,
            explorer: config.explorer,
            // expr_sampling: config.expr_sampling,
            device,
            n_opts: 0,
            _clip_reward: config.clip_reward,
            double_dqn: config.double_dqn,
            phantom: PhantomData,
        }
    }

    fn update_critic(&mut self, batch: R::Batch) -> f32 {
        let (obs, act, next_obs, reward, is_done) = batch.unpack();
        let obs = obs.into();
        let act = act.into().to(self.device);
        let next_obs = next_obs.into();
        let reward = Tensor::of_slice(&reward[..]).to(self.device);
        let is_done = Tensor::of_slice(&is_done[..]).to(self.device);

        // let obs = &batch.obs;
        // let a = batch.actions.to(self.device);
        // let r = batch.rewards.to(self.device);
        // let next_obs = batch.next_obs;
        // let not_done = batch.not_dones.to(self.device);
        // let ixs = batch.indices;
        // let ws = batch.ws;

        let pred = {
            let x = self.qnet.forward(&obs);
            x.gather(-1, &act, false).squeeze()
        };

        let tgt = no_grad(|| {
            let q = if self.double_dqn {
                let x = self.qnet.forward(&next_obs);
                let y = x.argmax(-1, false).unsqueeze(-1);
                self.qnet_tgt.forward(&next_obs).gather(-1, &y, false).squeeze()
            } else {
                let x = self.qnet_tgt.forward(&next_obs);
                let y = x.argmax(-1, false).unsqueeze(-1);
                x.gather(-1, &y, false).squeeze()
            };
            reward + (1 - is_done) * self.discount_factor * q
        });

        let loss = pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0);

        // let loss = if let Some(ws) = ws {
        //     // with PER
        //     let ixs = ixs.unwrap();
        //     let tderr = (pred - tgt).abs(); //.clip(0.0, 1.0)
        //     let eps = Tensor::from(1e-5).internal_cast_float(false);
        //     self.replay_buffer.update_priority(&ixs, &(&tderr + eps));
        //     (tderr * ws.to(self.device)).smooth_l1_loss(
        //         &Tensor::from(0f32).to(self.device),
        //         tch::Reduction::Mean,
        //         1.0,
        //     )
        // } else {
        //     // w/o PER
        //     pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0)
        // };

        self.qnet.backward_step(&loss);

        f32::from(loss)
    }

    fn opt_(&mut self, buffer: &mut R) -> Record {
        let mut loss_critic = 0f32;

        let beta = None;
        // #[allow(unused_variables)]
        // let beta = match &self.expr_sampling {
        //     ExperienceSampling::Uniform => 0f32,
        //     ExperienceSampling::TDerror {
        //         alpha,
        //         iw_scheduler,
        //     } => iw_scheduler.beta(self.n_opts),
        // };

        for _ in 0..self.n_updates_per_opt {
            let batch = buffer.batch(self.batch_size, beta).unwrap();
            loss_critic += self.update_critic(batch);
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
                a.argmax(-1, true)
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

// impl<E, Q, O, A> Agent<E> for DQN<E, Q, O, A>
// where
//     E: Env,
//     Q: SubModel<Output = Tensor>,
//     E::Obs: Into<Q::Input>,
//     E::Act: From<Tensor>,
//     O: TchBuffer<Item = E::Obs, SubBatch = Q::Input>,
//     A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
// {
//     fn train(&mut self) {
//         self.train = true;
//     }

//     fn eval(&mut self) {
//         self.train = false;
//     }

//     fn is_train(&self) -> bool {
//         self.train
//     }

//     fn opt(&mut self, buffer: &mut impl border_core::ReplayBufferBase) -> Option<Record> {
//         if buffer.len() >= self.min_transitions_warmup {
//             let batch = buffer.batch(self.batch_size, None).unwrap();
//             Some(self.opt_(buffer))
//         } else {
//             None
//         }
//     }

//     /// Update model parameters.
//     ///
//     /// When the return value is `Some(Record)`, it includes:
//     /// * `loss_critic`: Loss of critic
//     fn observe(&mut self, step: Step<E>) -> Option<Record> {
//         trace!("DQN::observe()");

//         // Check if doing optimization
//         let do_optimize = self.opt_interval_counter.do_optimize(&step.is_done)
//             && self.replay_buffer.len() + 1 >= self.min_transitions_warmup;

//         // Push transition to the replay buffer
//         self.push_transition(step);
//         trace!("Push transition");

//         // Do optimization
//         if do_optimize {
//             Some(self.opt())
//         } else {
//             None
//         }
//     }

//     fn save<T: AsRef<Path>>(&self, path: T) -> Result<()> {
//         // TODO: consider to rename the path if it already exists
//         fs::create_dir_all(&path)?;
//         self.qnet.save(&path.as_ref().join("qnet.pt").as_path())?;
//         self.qnet_tgt
//             .save(&path.as_ref().join("qnet_tgt.pt").as_path())?;
//         Ok(())
//     }

//     fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
//         self.qnet.load(&path.as_ref().join("qnet.pt").as_path())?;
//         self.qnet_tgt
//             .load(&path.as_ref().join("qnet_tgt.pt").as_path())?;
//         Ok(())
//     }
// }
