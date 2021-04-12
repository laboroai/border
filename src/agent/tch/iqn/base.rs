//! IQN agent implemented with tch-rs.
use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{Tensor, no_grad, Device};

use crate::{
    core::{
        Policy, Agent, Step, Env,
        record::{Record, RecordValue}
    },
    agent::{
        OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer, // TchBatch,
            model::Model1, util::{track, quantile_huber_loss},
            iqn::{IQNModel, IQNExplorer, model::{IQNAverage::*, average}},
        }
    }
};

#[allow(clippy::upper_case_acronyms)]
/// IQN agent implemented with tch-rs.
///
/// The type parameter `M` is a feature extractor, which takes
/// `M::Input` and returns feature vectors.
pub struct IQN<E, M, O, A> where
    E: Env,
    M: Model1 + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    pub(super) opt_interval_counter: OptIntervalCounter,
    pub(super) soft_update_interval: usize,
    pub(super) soft_update_counter: usize,
    pub(super) n_updates_per_opt: usize,
    pub(super) min_transitions_warmup: usize,
    pub(super) batch_size: usize,
    pub(super) feature: M,
    pub(super) feature_tgt: M,
    // pub(super) iqn: IQNModel,
    // pub(super) iqn_tgt: IQNModel,
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

// impl<E, M, O, A> IQN<E, M, O, A> where
//     E: Env,
//     M: Model1<Output=Tensor> + Clone,
//     E::Obs :Into<M::Input>,
//     E::Act :From<Tensor>,
//     O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
//     A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
// {
//     fn push_transition(&mut self, step: Step<E>) {
//         trace!("IQN::push_transition()");

//         let next_obs = step.obs;
//         let obs = self.prev_obs.replace(None).unwrap();
//         let reward = Tensor::of_slice(&step.reward[..]);
//         let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);
//         self.replay_buffer.push(
//             &obs,
//             &step.act,
//             &reward,
//             &next_obs,
//             &not_done,
//         );
//         let _ = self.prev_obs.replace(Some(next_obs));
//     }

//     fn update_critic(&mut self, batch: TchBatch<E, O, A>) -> f32 {
//         trace!("IQN::update_critic()");

//         let obs = batch.obs.to(self.device);
//         let a = batch.actions.to(self.device);
//         let r = batch.rewards.to(self.device);
//         let next_obs = batch.next_obs.to(self.device);
//         let not_done = batch.not_dones.to(self.device);
//         trace!("obs.shape      = {:?}", obs.size());
//         trace!("next_obs.shape = {:?}", next_obs.size());
//         trace!("a.shape        = {:?}", a.size());

//         debug_assert_eq!(r.size().as_slice(), &[self.batch_size as i64]);
//         debug_assert_eq!(not_done.size().as_slice(), &[self.batch_size as i64]);

//         let loss = {
//             let pred = {
//                 let a = a;

//                 // x.size() == [batch_size, n_actions, n_prob_samples]
//                 let x = self.iqn.forward(&obs);
//                 debug_assert_eq!(x.size().as_slice()[0], self.batch_size as i64);
//                 debug_assert_eq!(x.size().as_slice()[0], self.n_prob_samples as i64);

//                 // x.size() == [batch_size, 1, n_prob_samples]
//                 let x = x.gather(1, &a, false).unsqueeze(1);
//                 debug_assert_eq!(
//                     x.size().as_slice(),
//                     &[self.batch_size as i64, 1, self.n_prob_samples as i64]
//                 );
//                 x
//             };
//             let tgt = no_grad(|| {
//                 let x = self.iqn_tgt.forward(&next_obs);

//                 // Takes average over quantile samples (tau_i), then takes argmax as actions
//                 let y = x.copy().mean1(&[-1], false, tch::Kind::Float);
//                 let ixs_act = y.argmax(-1, false).unsqueeze(-1);

//                 // x.size() == [batch_size, n_prob_samples]
//                 let x = x.gather(1, &ixs_act, false);
//                 debug_assert_eq!(
//                     x.size().as_slice(),
//                     &[self.batch_size as i64, self.n_prob_samples as i64]
//                 );

//                 // x.size() == [batch_size, n_prob_samples, 1]
//                 let x = r + not_done * self.discount_factor * x;
//                 debug_assert_eq!(
//                     x.size().as_slice(),
//                     &[self.batch_size as i64, self.n_prob_samples as i64, 1]
//                 );
//                 x
//             });
//             quantile_huber_loss(&(pred - tgt))
//         };
//         self.iqn.backward_step(&loss);

//         f32::from(loss)
//     }

//     fn soft_update(&mut self) {
//         trace!("IQN::soft_update()");
//         // track(&mut self.iqn_tgt, &mut self.iqn, self.tau);
//         track(&mut self.feature_tgt, &mut self.feature, self.tau);
//     }
// }

impl<E, M, O, A> Policy<E> for IQN<E, M, O, A> where
    E: Env,
    M: Model1<Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = M::Output>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let psi = self.feature.forward(&obs);

        let a = no_grad(|| {
            if self.train {
                let iqn = &self.iqn;
                let device = self.device;
                let q_fn = || {
                    let a = average(&psi, iqn, Uniform10, device);
                    a.argmax(-1, true)
                };
                let n_procs = psi.size()[0];
                let shape = (n_procs as u32, self.iqn.out_dim as u32);
                match &mut self.explorer {
                    IQNExplorer::EpsilonGreedy(egreedy) => egreedy.action(shape, q_fn),
                }
            } else {
                let a = average(&psi, &self.iqn, Uniform10, self.device);
                a.argmax(-1, true)
            }
        });
        a.into()
    }
}

// impl<E, M, O, A> Agent<E> for IQN<E, M, O, A> where
//     E: Env,
//     M: Model1<Output=Tensor> + Clone,
//     E::Obs :Into<M::Input>,
//     E::Act :From<Tensor>,
//     O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
//     A: TchBuffer<Item = E::Act, SubBatch = M::Output>,
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

//     fn push_obs(&self, obs: &E::Obs) {
//         self.prev_obs.replace(Some(obs.clone()));
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
//             let mut loss_critic = 0f32;

//             for _ in 0..self.n_updates_per_opt {
//                 let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
//                 trace!("Sample random batch");

//                 loss_critic += self.update_critic(batch);
//             };

//             self.soft_update_counter += 1;
//             if self.soft_update_counter == self.soft_update_interval {
//                 self.soft_update_counter = 0;
//                 self.soft_update();
//                 trace!("Update target network");
//             }

//             loss_critic /= self.n_updates_per_opt as f32;

//             Some(Record::from_slice(&[
//                 ("loss_critic", RecordValue::Scalar(loss_critic)),
//             ]))
//         }
//         else {
//             None
//         }
//     }

//     fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
//         // TODO: consider to rename the path if it already exists
//         fs::create_dir_all(&path)?;
//         self.qnet.save(&path.as_ref().join("qnet.pt").as_path())?;
//         self.qnet_tgt.save(&path.as_ref().join("qnet_tgt.pt").as_path())?;
//         Ok(())
//     }

//     fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
//         self.qnet.load(&path.as_ref().join("qnet.pt").as_path())?;
//         self.qnet_tgt.load(&path.as_ref().join("qnet_tgt.pt").as_path())?;
//         Ok(())
//     }
// }
