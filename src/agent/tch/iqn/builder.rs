//! Builder of IQN agent
use std::{cell::RefCell, marker::PhantomData};
use tch::Tensor;

use crate::{
    core::Env,
    agent::{
        OptInterval, OptIntervalCounter,
        tch::{
            ReplayBuffer, TchBuffer,
            model::Model1,
            dqn::{DQN, explorer::{DQNExplorer, Softmax}}
        }
    }
};

pub struct IQNBuilder<E, M, O, A> where
    E: Env,
    M: Model1<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    opt_interval_counter: OptIntervalCounter,
    soft_update_interval: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    train: bool,
    discount_factor: f64,
    tau: f64,
    explorer: DQNExplorer,
    phantom: PhantomData<(E, M, O, A)>,
}
