//! IQN agent implemented with tch-rs.
// use log::trace;
// use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use std::{cell::RefCell, marker::PhantomData};
// use tch::{no_grad, Tensor};
use tch::{Tensor};

use crate::{
    core::{
        // Policy, Agent, Step, Env,
        Env,
        // record::{Record, RecordValue}
    },
    agent::{
        OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer, // TchBatch,
            model::Model1, //util::{track, quantile_huber_loss},
            dqn::explorer::DQNExplorer, // iqn::IQN,
        }
    }
};

/// IQN agent implemented with tch-rs.
pub struct IQNAgent<E, M, O, A> where
    E: Env,
    M: Model1 + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    pub(self) opt_interval_counter: OptIntervalCounter,
    pub(self) soft_update_interval: usize,
    pub(self) soft_update_counter: usize,
    pub(self) n_updates_per_opt: usize,
    pub(self) min_transitions_warmup: usize,
    pub(self) batch_size: usize,
    pub(self) iqn: M,
    pub(self) iqn_tgt: M,
    pub(self) train: bool,
    pub(self) phantom: PhantomData<E>,
    pub(self) prev_obs: RefCell<Option<E::Obs>>,
    pub(self) replay_buffer: ReplayBuffer<E, O, A>,
    pub(self) discount_factor: f64,
    pub(self) tau: f64,
    pub(self) n_prob_samples: usize,
    pub(self) explorer: DQNExplorer,
    pub(self) device: tch::Device,
}

// impl<E, M, O, A> IQNAgent<E, M, O, A> where
//     E: Env,
//     M: IQN<Input=Tensor, Output=Tensor> + Clone,
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
//         track(&mut self.iqn_tgt, &mut self.iqn, self.tau);
//     }
// }
