//! Asynchronous trainer with parallel sampling processes.
//!
//! The code might look like below.
//!
//! ```
//! # use serde::{Deserialize, Serialize};
//! # use border_core::test::{
//! #     TestAgent, TestAgentConfig, TestEnv, TestObs, TestObsBatch,
//! #     TestAct, TestActBatch
//! # };
//! # use border_async_trainer::{
//! #     //test::{TestAgent, TestAgentConfig, TestEnv},
//! #     ActorManager, ActorManagerConfig, AsyncTrainer, AsyncTrainerConfig,
//! # };
//! # use border_core::{
//! #     generic_replay_buffer::{
//! #         SimpleReplayBuffer, SimpleReplayBufferConfig,
//! #         SimpleStepProcessorConfig, SimpleStepProcessor
//! #     },
//! #     record::{AggregateRecorder, NullRecorder}, DefaultEvaluator,
//! # };
//! #
//! # use std::path::Path;
//! #
//! # fn agent_config() -> TestAgentConfig {
//! #     TestAgentConfig
//! # }
//! #
//! # fn env_config() -> usize {
//! #     0
//! # }
//!
//! type Env = TestEnv;
//! type ObsBatch = TestObsBatch;
//! type ActBatch = TestActBatch;
//! type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! type StepProcessor = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//!
//! // Create a new agent by wrapping the existing agent in order to implement SyncModel.
//! struct TestAgent2(TestAgent);
//!
//! impl border_core::Configurable for TestAgent2 {
//!     type Config = TestAgentConfig;
//!
//!     fn build(config: Self::Config) -> Self {
//!         Self(TestAgent::build(config))
//!     }
//! }
//!
//! impl border_core::Agent<Env, ReplayBuffer> for TestAgent2 {
//!     // Boilerplate code to delegate the method calls to the inner agent.
//!     fn train(&mut self) {
//!         self.0.train();
//!      }
//!
//!      // For other methods ...
//! #     fn is_train(&self) -> bool {
//! #         self.0.is_train()
//! #     }
//! #
//! #     fn eval(&mut self) {
//! #         self.0.eval();
//! #     }
//! #
//! #     fn opt_with_record(&mut self, buffer: &mut ReplayBuffer) -> border_core::record::Record {
//! #         self.0.opt_with_record(buffer)
//! #     }
//! #
//! #     fn save_params(&self, path: &Path) -> anyhow::Result<()> {
//! #         self.0.save_params(path)
//! #     }
//! #
//! #     fn load_params(&mut self, path: &Path) -> anyhow::Result<()> {
//! #         self.0.load_params(path)
//! #     }
//! #
//! #     fn opt(&mut self, buffer: &mut ReplayBuffer) {
//! #         self.0.opt_with_record(buffer);
//! #     }
//! #
//! #     fn as_any_ref(&self) -> &dyn std::any::Any {
//! #         self
//! #     }
//! #
//! #     fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
//! #         self
//! #     }
//! }
//!
//! impl border_core::Policy<Env> for TestAgent2 {
//!       // Boilerplate code to delegate the method calls to the inner agent.
//!       // ...
//! #     fn sample(&mut self, obs: &TestObs) -> TestAct {
//! #         self.0.sample(obs)
//! #     }
//! }
//!
//! impl border_async_trainer::SyncModel for TestAgent2{
//!     // Self::ModelInfo shold include the model parameters.
//!     type ModelInfo = usize;
//!
//!
//!     fn model_info(&self) -> (usize, Self::ModelInfo) {
//!         // Extracts the model parameters and returns them as Self::ModelInfo.
//!         // The first element of the tuple is the number of optimization steps.
//!         (0, 0)
//!     }
//!
//!     fn sync_model(&mut self, _model_info: &Self::ModelInfo) {
//!         // implements synchronization of the model based on the _model_info
//!     }
//! }
//!
//! let agent_configs: Vec<_> = vec![agent_config()];
//! let env_config_train = env_config();
//! let env_config_eval = env_config();
//! let replay_buffer_config = SimpleReplayBufferConfig::default();
//! let step_proc_config = SimpleStepProcessorConfig::default();
//! let actor_man_config = ActorManagerConfig::default();
//! let async_trainer_config = AsyncTrainerConfig::default();
//! let mut recorder: Box<dyn AggregateRecorder> = Box::new(NullRecorder {});
//! let mut evaluator = DefaultEvaluator::<TestEnv>::new(&env_config_eval, 0, 1).unwrap();
//!
//! border_async_trainer::util::train_async::<TestAgent2, _, _, StepProcessor>(
//!     &agent_config(),
//!     &agent_configs,
//!     &env_config_train,
//!     &env_config_eval,
//!     &step_proc_config,
//!     &replay_buffer_config,
//!     &actor_man_config,
//!     &async_trainer_config,
//!     &mut recorder,
//!     &mut evaluator,
//! );
//! ```
//!
//! Training process consists of the following two components:
//!
//! * [`ActorManager`] manages [`Actor`]s, each of which runs a thread for interacting
//!   [`Agent`] and [`Env`] and taking samples. Those samples will be sent to
//!   the replay buffer in [`AsyncTrainer`].
//! * [`AsyncTrainer`] is responsible for training of an agent. It also runs a thread
//!   for pushing samples from [`ActorManager`] into a replay buffer.
//!
//! The `Agent` must implement [`SyncModel`] trait in order to synchronize the model of
//! the agent in [`Actor`] with the trained agent in [`AsyncTrainer`]. The trait has
//! the ability to import and export the information of the model as
//! [`SyncModel`]`::ModelInfo`.
//!
//! The `Agent` in [`AsyncTrainer`] is responsible for training, typically with a GPU,
//! while the `Agent`s in [`Actor`]s in [`ActorManager`] is responsible for sampling
//! using CPU.
//!
//! Both [`AsyncTrainer`] and [`ActorManager`] are running in the same machine and
//! communicate by channels.
//!
//! [`Agent`]: border_core::Agent
//! [`Env`]: border_core::Env
mod actor;
mod actor_manager;
mod async_trainer;
mod error;
mod messages;
mod replay_buffer_proxy;
mod sync_model;
pub mod util;

pub use actor::{actor_stats_fmt, Actor, ActorStat};
pub use actor_manager::{ActorManager, ActorManagerConfig};
pub use async_trainer::{AsyncTrainStat, AsyncTrainer, AsyncTrainerConfig};
pub use error::BorderAsyncTrainerError;
pub use messages::PushedItemMessage;
pub use replay_buffer_proxy::{ReplayBufferProxy, ReplayBufferProxyConfig};
pub use sync_model::SyncModel;

/// Agent and Env for testing.
#[cfg(test)]
pub mod test {
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    /// Obs for testing.
    #[derive(Clone, Debug)]
    pub struct TestObs {
        obs: usize,
    }

    impl border_core::Obs for TestObs {
        fn len(&self) -> usize {
            1
        }
    }

    /// Batch of obs for testing.
    pub struct TestObsBatch {
        obs: Vec<usize>,
    }

    impl border_core::generic_replay_buffer::BatchBase for TestObsBatch {
        fn new(capacity: usize) -> Self {
            Self {
                obs: vec![0; capacity],
            }
        }

        fn push(&mut self, i: usize, data: Self) {
            self.obs[i] = data.obs[0];
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let obs = ixs.iter().map(|ix| self.obs[*ix]).collect();
            Self { obs }
        }
    }

    impl From<TestObs> for TestObsBatch {
        fn from(obs: TestObs) -> Self {
            Self { obs: vec![obs.obs] }
        }
    }

    /// Act for testing.
    #[derive(Clone, Debug)]
    pub struct TestAct {
        act: usize,
    }

    impl border_core::Act for TestAct {}

    /// Batch of act for testing.
    pub struct TestActBatch {
        act: Vec<usize>,
    }

    impl From<TestAct> for TestActBatch {
        fn from(act: TestAct) -> Self {
            Self { act: vec![act.act] }
        }
    }

    impl border_core::generic_replay_buffer::BatchBase for TestActBatch {
        fn new(capacity: usize) -> Self {
            Self {
                act: vec![0; capacity],
            }
        }

        fn push(&mut self, i: usize, data: Self) {
            self.act[i] = data.act[0];
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let act = ixs.iter().map(|ix| self.act[*ix]).collect();
            Self { act }
        }
    }

    /// Info for testing.
    pub struct TestInfo {}

    impl border_core::Info for TestInfo {}

    /// Environment for testing.
    pub struct TestEnv {
        state_init: usize,
        state: usize,
    }

    impl border_core::Env for TestEnv {
        type Config = usize;
        type Obs = TestObs;
        type Act = TestAct;
        type Info = TestInfo;

        fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> anyhow::Result<Self::Obs> {
            self.state = self.state_init;
            Ok(TestObs { obs: self.state })
        }

        fn reset_with_index(&mut self, _ix: usize) -> anyhow::Result<Self::Obs> {
            self.state = self.state_init;
            Ok(TestObs { obs: self.state })
        }

        fn step_with_reset(
            &mut self,
            a: &Self::Act,
        ) -> (border_core::Step<Self>, border_core::record::Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = border_core::Step {
                obs: TestObs { obs: self.state },
                act: a.clone(),
                reward: vec![0.0],
                is_terminated: vec![0],
                is_truncated: vec![0],
                info: TestInfo {},
                init_obs: Some(TestObs {
                    obs: self.state_init,
                }),
            };
            return (step, border_core::record::Record::empty());
        }

        fn step(&mut self, a: &Self::Act) -> (border_core::Step<Self>, border_core::record::Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = border_core::Step {
                obs: TestObs { obs: self.state },
                act: a.clone(),
                reward: vec![0.0],
                is_terminated: vec![0],
                is_truncated: vec![0],
                info: TestInfo {},
                init_obs: Some(TestObs {
                    obs: self.state_init,
                }),
            };
            return (step, border_core::record::Record::empty());
        }

        fn build(config: &Self::Config, _seed: i64) -> anyhow::Result<Self>
        where
            Self: Sized,
        {
            Ok(Self {
                state_init: *config,
                state: 0,
            })
        }
    }

    type ReplayBuffer =
        border_core::generic_replay_buffer::SimpleReplayBuffer<TestObsBatch, TestActBatch>;

    /// Agent for testing.
    pub struct TestAgent {}

    #[derive(Clone, Deserialize, Serialize)]
    /// Config of agent for testing.
    pub struct TestAgentConfig;

    impl border_core::Agent<TestEnv, ReplayBuffer> for TestAgent {
        fn train(&mut self) {}

        fn is_train(&self) -> bool {
            false
        }

        fn eval(&mut self) {}

        fn opt_with_record(&mut self, _buffer: &mut ReplayBuffer) -> border_core::record::Record {
            border_core::record::Record::empty()
        }

        fn save_params(&self, _path: &Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn load_params(&mut self, _path: &Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn as_any_ref(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl border_core::Policy<TestEnv> for TestAgent {
        fn sample(&mut self, _obs: &TestObs) -> TestAct {
            TestAct { act: 1 }
        }
    }

    impl border_core::Configurable for TestAgent {
        type Config = TestAgentConfig;

        fn build(_config: Self::Config) -> Self {
            Self {}
        }
    }

    impl crate::SyncModel for TestAgent {
        type ModelInfo = usize;

        fn model_info(&self) -> (usize, Self::ModelInfo) {
            (0, 0)
        }

        fn sync_model(&mut self, _model_info: &Self::ModelInfo) {
            // nothing to do
        }
    }
}
