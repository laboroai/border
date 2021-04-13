//! IQN agent implemented with tch-rs.
use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{Tensor, no_grad, Device};

use crate::{
    core::{
        Policy, Agent, Step, Env, Obs,
        record::{Record, RecordValue}
    },
    agent::{
        OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer, model::ModelBase, TchBatch,
            iqn::{IQNModel, IQNExplorer, model::{IQNSample, average}},
            util::{FeatureExtractor, quantile_huber_loss, track}
        }
    },
};

#[allow(clippy::upper_case_acronyms)]
/// IQN agent implemented with tch-rs.
///
/// The type parameter `M` is a feature extractor, which takes
/// `M::Input` and returns feature vectors.
pub struct IQN<E, F, O, A> where
    E: Env,
    F: FeatureExtractor,
    E::Obs :Into<F::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    pub(super) opt_interval_counter: OptIntervalCounter,
    pub(super) soft_update_interval: usize,
    pub(super) soft_update_counter: usize,
    pub(super) n_updates_per_opt: usize,
    pub(super) min_transitions_warmup: usize,
    pub(super) batch_size: usize,
    pub(super) iqn: IQNModel<F>,
    pub(super) iqn_tgt: IQNModel<F>,
    pub(super) train: bool,
    pub(super) phantom: PhantomData<E>,
    pub(super) prev_obs: RefCell<Option<E::Obs>>,
    pub(super) replay_buffer: ReplayBuffer<E, O, A>,
    pub(super) discount_factor: f64,
    pub(super) tau: f64,
    pub(super) n_prob_samples: usize,
    pub(super) explorer: IQNExplorer,
    pub(super) device: Device,
}

impl<E, F, O, A> IQN<E, F, O, A> where
    E: Env,
    F: FeatureExtractor,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn push_transition(&mut self, step: Step<E>) {
        trace!("IQN::push_transition()");

        let next_obs = step.obs;
        let obs = self.prev_obs.replace(None).unwrap();
        let reward = Tensor::of_slice(&step.reward[..]);
        let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);
        self.replay_buffer.push(
            &obs,
            &step.act,
            &reward,
            &next_obs,
            &not_done,
        );
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn update_critic(&mut self, batch: TchBatch<E, O, A>) -> f32 {
        trace!("IQN::update_critic()");

        let obs = batch.obs;
        let a = batch.actions.to(self.device);
        let r = batch.rewards.to(self.device);
        let next_obs = batch.next_obs;
        let not_done = batch.not_dones.to(self.device);
        trace!("a.shape        = {:?}", a.size());

        debug_assert_eq!(r.size().as_slice(), &[self.batch_size as i64]);
        debug_assert_eq!(not_done.size().as_slice(), &[self.batch_size as i64]);

        let loss = {
            let pred = {
                let a = a;

                // x.size() == [batch_size, n_actions, n_prob_samples]
                let tau = IQNSample::Uniform10.sample();
                let x = self.iqn.forward(&obs, &tau);
                debug_assert_eq!(x.size().as_slice()[0], self.batch_size as i64);
                debug_assert_eq!(x.size().as_slice()[0], self.n_prob_samples as i64);

                // x.size() == [batch_size, 1, n_prob_samples]
                let x = x.gather(1, &a, false).unsqueeze(1);
                debug_assert_eq!(
                    x.size().as_slice(),
                    &[self.batch_size as i64, 1, self.n_prob_samples as i64]
                );
                x
            };
            let tau_pred = IQNSample::Uniform10.sample();
            let tgt = no_grad(|| {
                let x = self.iqn_tgt.forward(&next_obs, &tau_pred);

                // Takes average over quantile samples (tau_i), then takes argmax as actions
                let y = x.copy().mean1(&[-1], false, tch::Kind::Float);
                let ixs_act = y.argmax(-1, false).unsqueeze(-1);

                // x.size() == [batch_size, n_prob_samples]
                let x = x.gather(1, &ixs_act, false);
                debug_assert_eq!(
                    x.size().as_slice(),
                    &[self.batch_size as i64, self.n_prob_samples as i64]
                );

                // x.size() == [batch_size, n_prob_samples, 1]
                let x = r + not_done * self.discount_factor * x;
                debug_assert_eq!(
                    x.size().as_slice(),
                    &[self.batch_size as i64, self.n_prob_samples as i64, 1]
                );
                x
            });
            quantile_huber_loss(&(pred - tgt), &tau_pred)
        };
        self.iqn.backward_step(&loss);

        f32::from(loss)
    }

    fn soft_update(&mut self) {
        trace!("IQN::soft_update()");
        track(&mut self.iqn_tgt, &mut self.iqn, self.tau);
    }
}

impl<E, F, O, A> Policy<E> for IQN<E, F, O, A> where
    E: Env,
    F: FeatureExtractor,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let a = no_grad(|| {
            if self.train {
                let iqn = &self.iqn;
                let device = self.device;
                let n_procs = obs.n_procs();
                let obs = obs.clone().into();
                let q_fn = || {
                    let a = average(&obs, iqn, IQNSample::Uniform10, device);
                    a.argmax(-1, true)
                };
                let shape = (n_procs as u32, self.iqn.out_dim as u32);
                match &mut self.explorer {
                    IQNExplorer::EpsilonGreedy(egreedy) => egreedy.action(shape, q_fn),
                }
            } else {
                let obs = obs.clone().into();
                let a = average(&obs, &self.iqn, IQNSample::Uniform10, self.device);
                a.argmax(-1, true)
            }
        });
        a.into()
    }
}

impl<E, F, O, A> Agent<E> for IQN<E, F, O, A> where
    E: Env,
    F: FeatureExtractor,
    E::Obs :Into<F::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = F::Input>,
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
            let mut loss_critic = 0f32;

            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                trace!("Sample random batch");

                loss_critic += self.update_critic(batch);
            };

            self.soft_update_counter += 1;
            if self.soft_update_counter == self.soft_update_interval {
                self.soft_update_counter = 0;
                self.soft_update();
                trace!("Update target network");
            }

            loss_critic /= self.n_updates_per_opt as f32;

            Some(Record::from_slice(&[
                ("loss_critic", RecordValue::Scalar(loss_critic)),
            ]))
        }
        else {
            None
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        self.iqn.save(&path.as_ref().join("iqn.pt").as_path())?;
        self.iqn_tgt.save(&path.as_ref().join("iqn_tgt.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.iqn.load(&path.as_ref().join("iqn.pt").as_path())?;
        self.iqn_tgt.load(&path.as_ref().join("iqn_tgt.pt").as_path())?;
        Ok(())
    }
}
