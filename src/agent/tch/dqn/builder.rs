//! Constructs DQN agent.
// use log::trace;
use std::{
    default::Default,
    cell::RefCell,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use tch::Tensor;

use crate::{
    agent::{
        tch::{
            dqn::{
                explorer::{DQNExplorer, Softmax},
                DQN,
            },
            model::Model1,
            ReplayBuffer, TchBuffer,
        },
        OptInterval, OptIntervalCounter,
    },
    core::Env,
};

#[allow(clippy::upper_case_acronyms)]
/// Constructs [DQN].
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
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

impl<E, M, O, A> Default for DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    fn default() -> Self {
        Self {
            opt_interval_counter: OptInterval::Steps(1).counter(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            train: false,
            explorer: DQNExplorer::Softmax(Softmax::new()),
            phantom: PhantomData,
        }
    }
}

impl<E, M, O, A> DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    /// Constructs DQN builder with default parameters.
    pub fn new() -> Self {
        Self {
            opt_interval_counter: OptInterval::Steps(1).counter(),
            soft_update_interval: 1,
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            train: false,
            explorer: DQNExplorer::Softmax(Softmax::new()),
            phantom: PhantomData,
        }
    }
}

impl<E, M, O, A> DQNBuilder<E, M, O, A>
where
    E: Env,
    M: Model1<Input = Tensor, Output = Tensor> + Clone,
    E::Obs: Into<M::Input>,
    E::Act: From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    /// Set optimization interval.
    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval_counter = v.counter();
        self
    }

    /// Set soft update interval.
    pub fn soft_update_interval(mut self, v: usize) -> Self {
        self.soft_update_interval = v;
        self
    }

    /// Set numper of parameter update steps per optimization step.
    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    /// Interval before starting optimization.
    pub fn min_transitions_warmup(mut self, v: usize) -> Self {
        self.min_transitions_warmup = v;
        self
    }

    /// Batch size.
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    /// Discount factor.
    pub fn discount_factor(mut self, v: f64) -> Self {
        self.discount_factor = v;
        self
    }

    /// Soft update coefficient.
    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    /// Set explorer.
    pub fn explorer(mut self, v: DQNExplorer) -> DQNBuilder<E, M, O, A> where {
        self.explorer = v;
        self
    }

    /// Constructs [TrainerBuilder] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [TrainerBuilder].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    /// Constructs DQN.
    pub fn build(
        self,
        qnet: M,
        replay_buffer: ReplayBuffer<E, O, A>,
        device: tch::Device,
    ) -> DQN<E, M, O, A> {
        let qnet_tgt = qnet.clone();

        DQN {
            qnet,
            qnet_tgt,
            replay_buffer,
            prev_obs: RefCell::new(None),
            opt_interval_counter: self.opt_interval_counter,
            soft_update_interval: self.soft_update_interval,
            soft_update_counter: 0,
            n_updates_per_opt: self.n_updates_per_opt,
            min_transitions_warmup: self.min_transitions_warmup,
            batch_size: self.batch_size,
            discount_factor: self.discount_factor,
            tau: self.tau,
            train: self.train,
            explorer: self.explorer,
            device,
            phantom: PhantomData,
        }
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use tempdir::TempDir;

//     use crate::{
//         agent::{
//             tch::{
//                 dqn::explorer::{DQNExplorer, EpsilonGreedy},
//                 model::Model1_1,
//                 DQNBuilder, ReplayBuffer,
//             },
//             OptInterval,
//         },
//         core::{
//             record::{BufferedRecorder, Record, TensorboardRecorder},
//             util, Agent, TrainerBuilder,
//         },
//         env::py_gym_env::{
//             act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
//             obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
//             tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
//             PyGymEnv, Shape,
//         },
//     };

//     const DIM_OBS: usize = 4;

//     #[derive(Debug, Clone)]
//     struct ObsShape {}

//     impl Shape for ObsShape {
//         fn shape() -> &'static [usize] {
//             &[DIM_OBS]
//         }
//     }

//     type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64, f32>;
//     type ActFilter = PyGymEnvDiscreteActRawFilter;
//     type Obs = PyGymEnvObs<ObsShape, f64, f32>;
//     type Act = PyGymEnvDiscreteAct;
//     type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

//     #[test]
//     fn test_serde_dqn_builder() -> Result<()> {
//         let builder = DQNBuilder::<Env, Model1_1, Obs, Act>::default();
//         Ok(())
//     }
// }
