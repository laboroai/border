//! Constructs DQN agent.
// use log::trace;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    default::Default,
    fs::File,
    io::{BufReader, Write},
    marker::PhantomData,
    path::Path,
};
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
pub struct DQNBuilder {
    opt_interval_counter: OptIntervalCounter,
    soft_update_interval: usize,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    train: bool,
    discount_factor: f64,
    tau: f64,
    replay_burffer_capacity: usize,
    explorer: DQNExplorer,
}

impl Default for DQNBuilder {
    /// Constructs DQN builder with default parameters.
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
            replay_burffer_capacity: 100,
            explorer: DQNExplorer::Softmax(Softmax::new()),
        }
    }
}

// impl DQNBuilder {
//     /// Constructs DQN builder with default parameters.
//     pub fn new() -> Self {
//         Self {
//             opt_interval_counter: OptInterval::Steps(1).counter(),
//             soft_update_interval: 1,
//             n_updates_per_opt: 1,
//             min_transitions_warmup: 1,
//             batch_size: 1,
//             discount_factor: 0.99,
//             tau: 0.005,
//             train: false,
//             explorer: DQNExplorer::Softmax(Softmax::new()),
//         }
//     }
// }

impl DQNBuilder {
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
    pub fn explorer(mut self, v: DQNExplorer) -> DQNBuilder {
        self.explorer = v;
        self
    }

    /// Replay buffer capacity.
    pub fn replay_burffer_capacity(mut self, v: usize) -> DQNBuilder {
        self.replay_burffer_capacity = v;
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

    /// Constructs DQN agent.
    ///
    /// This is used with non-vectorized environments.
    pub fn build<E, M, O, A>(self, qnet: M, device: tch::Device) -> DQN<E, M, O, A>
    where
        E: Env,
        M: Model1<Input = Tensor, Output = Tensor> + Clone,
        E::Obs: Into<M::Input>,
        E::Act: From<Tensor>,
        O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
        A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
    {
        let qnet_tgt = qnet.clone();
        let replay_buffer = ReplayBuffer::new(self.replay_burffer_capacity, 1);

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

    /// Constructs DQN agent with the given replay buffer.
    pub fn build_with_replay_buffer<E, M, O, A>(
        self,
        qnet: M,
        replay_buffer: ReplayBuffer<E, O, A>,
        device: tch::Device,
    ) -> DQN<E, M, O, A>
    where
        E: Env,
        M: Model1<Input = Tensor, Output = Tensor> + Clone,
        E::Obs: Into<M::Input>,
        E::Act: From<Tensor>,
        O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
        A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
    {
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

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    use crate::agent::{
        tch::{dqn::explorer::EpsilonGreedy, DQNBuilder},
        OptInterval,
    };

    #[test]
    fn test_serde_dqn_builder() -> Result<()> {
        let builder = DQNBuilder::default()
            .opt_interval(OptInterval::Steps(50))
            .n_updates_per_opt(1)
            .min_transitions_warmup(100)
            .batch_size(32)
            .discount_factor(0.99)
            .tau(0.005)
            .explorer(EpsilonGreedy::with_final_step(1000));

        let dir = TempDir::new("dqn_builder")?;
        let path = dir.path().join("dqn_builder.yaml");
        println!("{:?}", path);

        builder.save(&path)?;
        let builder_ = DQNBuilder::load(&path)?;
        assert_eq!(builder, builder_);

        let yaml = serde_yaml::to_string(&builder)?;
        println!("{}", yaml);

        Ok(())
    }
}
