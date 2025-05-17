#![warn(missing_docs)]
//! Core components for reinforcement learning.
//!
//! # Observation and Action
//!
//! The [`Obs`] and [`Act`] traits provide abstractions for observations and actions in environments.
//!
//! # Environment
//!
//! The [`Env`] trait serves as the fundamental abstraction for environments. It defines four associated types:
//! `Config`, `Obs`, `Act`, and `Info`. The `Obs` and `Act` types represent concrete implementations of
//! environment observations and actions, respectively. These types must implement the [`Obs`] and [`Act`] traits.
//! Environments implementing [`Env`] generate [`Step<E: Env>`] objects at each interaction step through the
//! [`Env::step()`] method. The [`Info`] type stores additional information from each agent-environment interaction,
//! which may be empty (implemented as a zero-sized struct). The `Config` type represents environment configurations
//! and is used during environment construction.
//!
//! # Policy
//!
//! The [`Policy<E: Env>`] trait represents a decision-making policy. The [`Policy::sample()`] method takes an
//! `E::Obs` and generates an `E::Act`. Policies can be either probabilistic or deterministic, depending on the
//! implementation.
//!
//! # Agent
//!
//! In this crate, an [`Agent<E: Env, R: ReplayBufferBase>`] is defined as a trainable [`Policy<E: Env>`].
//! Agents operate in either training or evaluation mode. During training, the agent's policy may be probabilistic
//! to facilitate exploration, while in evaluation mode, it typically becomes deterministic.
//!
//! The [`Agent::opt()`] method executes a single optimization step. The specific implementation of an optimization
//! step varies between agents and may include multiple stochastic gradient descent steps. Training samples are
//! obtained from the [`ReplayBufferBase`].
//!
//! This trait also provides methods for saving and loading trained policy parameters to and from a directory.
//!
//! # Batch
//!
//! The [`TransitionBatch`] trait represents a batch of transitions in the form `(o_t, r_t, a_t, o_t+1)`.
//! This trait is used for training [`Agent`]s using reinforcement learning algorithms.
//!
//! # Replay Buffer and Experience Buffer
//!
//! The [`ReplayBufferBase`] trait provides an abstraction for replay buffers. Its associated type
//! [`ReplayBufferBase::Batch`] represents samples used for training [`Agent`]s. Agents must implement the
//! [`Agent::opt()`] method, where [`ReplayBufferBase::Batch`] must have appropriate type or trait bounds
//! for training the agent.
//!
//! While [`ReplayBufferBase`] focuses on generating training batches, the [`ExperienceBufferBase`] trait
//! handles sample storage. The [`ExperienceBufferBase::push()`] method stores samples of type
//! [`ExperienceBufferBase::Item`], typically obtained through environment interactions.
//!
//! ## Reference Implementation
//!
//! [`SimpleReplayBuffer<O, A>`] implements both [`ReplayBufferBase`] and [`ExperienceBufferBase`].
//! This type takes two parameters, `O` and `A`, representing observation and action types in the replay buffer.
//! Both `O` and `A` must implement [`BatchBase`], which provides sample storage functionality similar to `Vec<T>`.
//! The associated types `Item` and `Batch` are both [`GenericTransitionBatch`], representing sets of
//! `(o_t, r_t, a_t, o_t+1)` transitions.
//!
//! # Step Processor
//!
//! The [`StepProcessor`] trait plays a crucial role in the training pipeline by transforming environment
//! interactions into training samples. It processes [`Step<E: Env>`] objects, which contain the current
//! observation, action, reward, and next observation, into a format suitable for the replay buffer.
//!
//! The [`SimpleStepProcessor<E, O, A>`] is a concrete implementation that:
//! 1. Maintains the previous observation to construct complete transitions
//! 2. Converts environment-specific observations and actions (`E::Obs` and `E::Act`) into batch-compatible
//!    types (`O` and `A`) using the `From` trait
//! 3. Generates [`GenericTransitionBatch`] objects containing the complete transition
//!    `(o_t, a_t, o_t+1, r_t, is_terminated, is_truncated)`
//! 4. Handles episode termination by properly resetting the previous observation
//!
//! This processor is essential for implementing temporal difference learning algorithms, as it ensures
//! that transitions are properly formatted and stored in the replay buffer for training.
//!
//! [`SimpleStepProcessor<E, O, A>`] can be used with [`SimpleReplayBuffer<O, A>`]. It converts `E::Obs` and
//! `E::Act` into their respective [`BatchBase`] types and generates [`GenericTransitionBatch`]. This conversion
//! relies on the trait bounds `O: From<E::Obs>` and `A: From<E::Act>`.
//!
//! # Trainer
//!
//! The [`Trainer`] manages the training loop and related objects. A [`Trainer`] instance is configured with
//! training parameters such as the maximum number of optimization steps and the directory for saving agent
//! parameters during training. The [`Trainer::train`] method executes online training of an agent in an environment.
//! During the training loop, the agent interacts with the environment to collect samples and perform optimization
//! steps, while simultaneously recording various metrics.
//!
//! # Evaluator
//!
//! The [`Evaluator<E, P>`] trait is used to evaluate a policy's (`P`) performance in an environment (`E`).
//! An instance of this type is provided to the [`Trainer`] for policy evaluation during training.
//! [`DefaultEvaluator<E, P>`] serves as the default implementation of [`Evaluator<E, P>`]. This evaluator
//! runs the policy in the environment for a specified number of episodes. At the start of each episode,
//! the environment is reset using [`Env::reset_with_index()`] to control specific evaluation conditions.
//!
//! [`SimpleReplayBuffer`]: generic_replay_buffer::SimpleReplayBuffer
//! [`SimpleReplayBuffer<O, A>`]: generic_replay_buffer::SimpleReplayBuffer
//! [`BatchBase`]: generic_replay_buffer::BatchBase
//! [`GenericTransitionBatch`]: generic_replay_buffer::GenericTransitionBatch
//! [`SimpleStepProcessor`]: generic_replay_buffer::SimpleStepProcessor
//! [`SimpleStepProcessor<E, O, A>`]: generic_replay_buffer::SimpleStepProcessor
pub mod dummy;
pub mod error;
mod evaluator;
pub mod generic_replay_buffer;
pub mod record;

mod base;
pub use base::{
    Act, Agent, Configurable, Env, ExperienceBufferBase, Info, NullReplayBuffer, Obs, Policy,
    ReplayBufferBase, Step, StepProcessor, TransitionBatch,
};

mod trainer;
pub use evaluator::{DefaultEvaluator, Evaluator};
pub use trainer::{Sampler, Trainer, TrainerConfig};

// TODO: Consider to compile this module only for tests.
/// Agent and Env for testing.
pub mod test {
    use serde::{Deserialize, Serialize};

    /// Obs for testing.
    #[derive(Clone, Debug)]
    pub struct TestObs {
        obs: usize,
    }

    impl crate::Obs for TestObs {
        fn len(&self) -> usize {
            1
        }
    }

    /// Batch of obs for testing.
    pub struct TestObsBatch {
        obs: Vec<usize>,
    }

    impl crate::generic_replay_buffer::BatchBase for TestObsBatch {
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

    impl crate::Act for TestAct {}

    /// Batch of act for testing.
    pub struct TestActBatch {
        act: Vec<usize>,
    }

    impl From<TestAct> for TestActBatch {
        fn from(act: TestAct) -> Self {
            Self { act: vec![act.act] }
        }
    }

    impl crate::generic_replay_buffer::BatchBase for TestActBatch {
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

    impl crate::Info for TestInfo {}

    /// Environment for testing.
    pub struct TestEnv {
        state_init: usize,
        state: usize,
    }

    impl crate::Env for TestEnv {
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

        fn step_with_reset(&mut self, a: &Self::Act) -> (crate::Step<Self>, crate::record::Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = crate::Step {
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
            return (step, crate::record::Record::empty());
        }

        fn step(&mut self, a: &Self::Act) -> (crate::Step<Self>, crate::record::Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = crate::Step {
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
            return (step, crate::record::Record::empty());
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
        crate::generic_replay_buffer::SimpleReplayBuffer<TestObsBatch, TestActBatch>;

    /// Agent for testing.
    pub struct TestAgent {}

    #[derive(Clone, Deserialize, Serialize)]
    /// Config of agent for testing.
    pub struct TestAgentConfig;

    impl crate::Agent<TestEnv, ReplayBuffer> for TestAgent {
        fn train(&mut self) {}

        fn is_train(&self) -> bool {
            false
        }

        fn eval(&mut self) {}

        fn opt_with_record(&mut self, _buffer: &mut ReplayBuffer) -> crate::record::Record {
            crate::record::Record::empty()
        }

        fn save_params(&self, _path: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
            Ok(vec![])
        }

        fn load_params(&mut self, _path: &std::path::Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn as_any_ref(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    impl crate::Policy<TestEnv> for TestAgent {
        fn sample(&mut self, _obs: &TestObs) -> TestAct {
            TestAct { act: 1 }
        }
    }

    impl crate::Configurable for TestAgent {
        type Config = TestAgentConfig;

        fn build(_config: Self::Config) -> Self {
            Self {}
        }
    }
}
