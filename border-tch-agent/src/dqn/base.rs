//! DQN agent implemented with tch-rs.
use crate::{
    dqn::{explorer::DQNExplorer, model::DQNModel},
    model::{ModelBase, SubModel},
    replay_buffer::{ExperienceSampling, ReplayBuffer, TchBatch, TchBuffer},
    util::{track, OptIntervalCounter},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Env, Policy, Step,
};
use log::trace;
use std::{cell::RefCell, convert::TryInto, fs, marker::PhantomData, path::Path};
use tch::{no_grad, Device, Tensor};

#[allow(clippy::upper_case_acronyms)]
/// DQN agent implemented with tch-rs.
pub struct DQN<E, Q, O, A>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = Q::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    pub(in crate::dqn) opt_interval_counter: OptIntervalCounter,
    pub(in crate::dqn) soft_update_interval: usize,
    pub(in crate::dqn) soft_update_counter: usize,
    pub(in crate::dqn) n_updates_per_opt: usize,
    pub(in crate::dqn) min_transitions_warmup: usize,
    pub(in crate::dqn) batch_size: usize,
    pub(in crate::dqn) qnet: DQNModel<Q>,
    pub(in crate::dqn) qnet_tgt: DQNModel<Q>,
    pub(in crate::dqn) train: bool,
    pub(in crate::dqn) phantom: PhantomData<E>,
    pub(in crate::dqn) prev_obs: RefCell<Option<E::Obs>>,
    pub(in crate::dqn) replay_buffer: ReplayBuffer<E, O, A>,
    pub(in crate::dqn) discount_factor: f64,
    pub(in crate::dqn) tau: f64,
    pub(in crate::dqn) explorer: DQNExplorer,
    pub(in crate::dqn) expr_sampling: ExperienceSampling,
    pub(in crate::dqn) device: Device,
    pub(in crate::dqn) n_opts: usize,
    pub(in crate::dqn) double_dqn: bool,
    pub(in crate::dqn) clip_reward: Option<f64>,
}

impl<E, Q, O, A> DQN<E, Q, O, A>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = Q::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn push_transition(&mut self, step: Step<E>) {
        trace!("DQN::push_transition()");

        let next_obs = step.obs;
        let obs = self.prev_obs.replace(None).unwrap();
        let reward = Tensor::of_slice(&step.reward[..]);
        let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);

        let reward = if let Some(clip) = self.clip_reward {
            reward.clip(-clip, clip)
        } else {
            reward
        };

        self.replay_buffer
            .push(&obs, &step.act, &reward, &next_obs, &not_done);
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn update_critic(&mut self, batch: TchBatch<E, O, A>) -> f32 {
        trace!("DQN::update_critic()");

        let obs = &batch.obs;
        let a = batch.actions.to(self.device);
        let r = batch.rewards.to(self.device);
        let next_obs = batch.next_obs;
        let not_done = batch.not_dones.to(self.device);
        let ixs = batch.indices;
        let ws = batch.ws;

        let pred = {
            let a = a;
            let x = self.qnet.forward(&obs);
            x.gather(-1, &a, false)
        };
        let tgt = no_grad(|| {
            let x = if self.double_dqn {
                self.qnet.forward(&next_obs)
            } else {
                self.qnet_tgt.forward(&next_obs)
            };            
            let y = x.argmax(-1, false).unsqueeze(-1);
            let x = x.gather(-1, &y, false);
            r + not_done * self.discount_factor * x
        });
        let loss = if let Some(ws) = ws {
            // with PER
            let ixs = ixs.unwrap();
            let tderr = (pred - tgt).abs(); //.clip(0.0, 1.0)
            let eps = Tensor::from(1e-5).internal_cast_float(false);
            self.replay_buffer.update_priority(&ixs, &(&tderr + eps));
            (tderr * ws.to(self.device)).smooth_l1_loss(
                &Tensor::from(0f32).to(self.device),
                tch::Reduction::Mean,
                1.0,
            )
        } else {
            // w/o PER
            pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0)
        };

        self.qnet.backward_step(&loss);

        f32::from(loss)
    }

    fn soft_update(&mut self) {
        trace!("DQN::soft_update()");
        track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
    }

    fn opt(&mut self) -> Record {
        let mut loss_critic = 0f32;
        #[allow(unused_variables)]
        let beta = match &self.expr_sampling {
            ExperienceSampling::Uniform => 0f32,
            ExperienceSampling::TDerror {
                alpha,
                iw_scheduler,
            } => iw_scheduler.beta(self.n_opts),
        };

        for _ in 0..self.n_updates_per_opt {
            let batch = self
                .replay_buffer
                .random_batch(self.batch_size, beta)
                .unwrap();
            trace!("Sample random batch");

            loss_critic += self.update_critic(batch);
        }

        self.soft_update_counter += 1;
        if self.soft_update_counter == self.soft_update_interval {
            self.soft_update_counter = 0;
            self.soft_update();
            trace!("Update target network");
        }

        loss_critic /= self.n_updates_per_opt as f32;

        self.n_opts += 1;

        Record::from_slice(&[("loss_critic", RecordValue::Scalar(loss_critic))])
    }
}

impl<E, Q, O, A> Policy<E> for DQN<E, Q, O, A>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = Q::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
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

impl<E, Q, O, A> Agent<E> for DQN<E, Q, O, A>
where
    E: Env,
    Q: SubModel<Output = Tensor>,
    E::Obs: Into<Q::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = Q::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
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

    fn push_obs(&self, obs: &E::Obs) {
        self.prev_obs.replace(Some(obs.clone()));
    }

    /// Update model parameters.
    ///
    /// When the return value is `Some(Record)`, it includes:
    /// * `loss_critic`: Loss of critic
    fn observe(&mut self, step: Step<E>) -> Option<Record> {
        trace!("DQN::observe()");

        // Check if doing optimization
        let do_optimize = self.opt_interval_counter.do_optimize(&step.is_done)
            && self.replay_buffer.len() + 1 >= self.min_transitions_warmup;

        // Push transition to the replay buffer
        self.push_transition(step);
        trace!("Push transition");

        // Do optimization
        if do_optimize {
            Some(self.opt())
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
