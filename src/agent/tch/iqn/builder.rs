//! Builder of IQN agent
use std::{cell::RefCell, marker::PhantomData};
use tch::{Tensor, Device};

use crate::{
    core::Env,
    agent::{
        OptInterval, OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer,
            model::Model1,
            iqn::{IQN, IQNExplorer}
        }
    }
};

use super::IQNModel;

#[allow(clippy::new_without_default)]
pub struct IQNBuilder<E, M, O, A> where
    E: Env,
    M: Model1<Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = M::Output>,
{
    opt_interval_counter: OptIntervalCounter,
    soft_update_interval: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    train: bool,
    discount_factor: f64,
    tau: f64,
    explorer: IQNExplorer,
    phantom: PhantomData<(E, M, O, A)>,
}

// impl<E, M, O, A> Default for IQNBuilder<E, M, O, A> where
//     E: Env,
//     M: Model1<Output=Tensor> + Clone,
//     E::Obs :Into<M::Input>,
//     E::Act :From<Tensor>,
//     O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
//     A: TchBuffer<Item = E::Act, SubBatch = M::Output>,
// {
//     fn default() -> Self {
//         Self {
//             pub(super) opt_interval_counter: OptIntervalCounter,
//             pub(super) soft_update_interval: usize,
//             pub(super) soft_update_counter: usize,
//             pub(super) n_updates_per_opt: usize,
//             pub(super) min_transitions_warmup: usize,
//             pub(super) batch_size: usize,
//             pub(super) feature: M,
//             pub(super) feature_tgt: M,
//             pub(super) iqn: IQNModel,
//             pub(super) iqn_tgt: IQNModel,
//             pub(super) train: bool,
//             pub(super) phantom: PhantomData<E>,
//             pub(super) prev_obs: RefCell<Option<E::Obs>>,
//             pub(super) replay_buffer: ReplayBuffer<E, O, A>,
//             pub(super) discount_factor: f64,
//             pub(super) tau: f64,
//             pub(super) n_prob_samples: usize,
//             pub(super) explorer: IQNExplorer,
//             pub(super) device: Device,        
//         }
//     }
// }

// impl<E, M, O, A> IQNBuilder<E, M, O, A> where
//     E: Env,
//     M: Model1<Output=Tensor> + Clone,
//     E::Obs :Into<M::Input>,
//     E::Act :From<Tensor>,
//     O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
//     A: TchBuffer<Item = E::Act, SubBatch = M::Output>,
// {
//     pub fn build(self, feature: M, replay_buffer: ReplayBuffer<E, O, A>, device: Device) {
//         let p = feature.get_var_store().root();
//         let iqn = IQNModel::new(&p, self.in_dim, self.embed_dim, self.out_dim, device);
//         let feature_tgt = feature.clone();
//         let p = feature_tgt.get_var_store().root();
//         let iqn_tgt = 
//         IQN {
            
//         }
//     }
// }
