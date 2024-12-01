#![warn(missing_docs)]
//! Core components for reinforcement learning.
//!
//! # Observation and action
//!
//! [`Obs`] and [`Act`] traits are abstractions of observation and action in environments.
//! These traits can handle two or more samples for implementing vectorized environments,
//! although there is currently no implementation of vectorized environment.
//!
//! # Environment
//!
//! [`Env`] trait is an abstraction of environments. It has four associated types:
//! `Config`, `Obs`, `Act` and `Info`. `Obs` and `Act` are concrete types of
//! observation and action of the environment.
//! These types must implement [`Obs`] and [`Act`] traits, respectively.
//! The environment that implements [`Env`] generates [`Step<E: Env>`] object
//! at every environment interaction step with [`Env::step()`] method.
//! [`Info`] stores some information at every step of interactions of an agent and
//! the environment. It could be empty (zero-sized struct). `Config` represents
//! configurations of the environment and is used to build.
//!
//! # Policy
//!
//! [`Policy<E: Env>`] represents a policy. [`Policy::sample()`] takes `E::Obs` and
//! generates `E::Act`. It could be probabilistic or deterministic.
//!
//! # Agent
//!
//! In this crate, [`Agent<E: Env, R: ReplayBufferBase>`] is defined as trainable
//! [`Policy<E: Env>`]. It is in either training or evaluation mode. In training mode,
//! the agent's policy might be probabilistic for exploration, while in evaluation mode,
//! the policy might be deterministic.
//!
//! The [`Agent::opt()`] method performs a single optimization step. The definition of an
//! optimization step varies for each agent. It might be multiple stochastic gradient
//! steps in an optimization step. Samples for training are taken from
//! [`R: ReplayBufferBase`][`ReplayBufferBase`].
//!
//! This trait also has methods for saving/loading parameters of the trained policy
//! in a directory.
//!
//! # Batch
//!
//! [`TransitionBatch`] is a trait of a batch of transitions `(o_t, r_t, a_t, o_t+1)`.
//! This trait is used to train [`Agent`]s using an RL algorithm.
//!
//! # Replay buffer and experience buffer
//!
//! [`ReplayBufferBase`] trait is an abstraction of replay buffers.
//! One of the associated type [`ReplayBufferBase::Batch`] represents samples taken from
//! the buffer for training [`Agent`]s. Agents must implements [`Agent::opt()`] method,
//! where [`ReplayBufferBase::Batch`] has an appropriate type or trait bound(s) to train
//! the agent.
//!
//! As explained above, [`ReplayBufferBase`] trait has an ability to generates batches
//! of samples with which agents are trained. On the other hand, [`ExperienceBufferBase`]
//! trait has an ability to store samples. [`ExperienceBufferBase::push()`] is used to push
//! samples of type [`ExperienceBufferBase::Item`], which might be obtained via interaction
//! steps with an environment.
//!
//! ## A reference implementation
//!
//! [`SimpleReplayBuffer<O, A>`] implementats both [`ReplayBufferBase`] and [`ExperienceBufferBase`].
//! This type has two parameters `O` and `A`, which are representation of
//! observation and action in the replay buffer. `O` and `A` must implement
//! [`BatchBase`], which has the functionality of storing samples, like `Vec<T>`,
//! for observation and action. The associated types `Item` and `Batch`
//! are the same type, [`GenericTransitionBatch`], representing sets of `(o_t, r_t, a_t, o_t+1)`.
//!
//! [`SimpleStepProcessor<E, O, A>`] might be used with [`SimpleReplayBuffer<O, A>`].
//! It converts `E::Obs` and `E::Act` into [`BatchBase`]s of respective types
//! and generates [`GenericTransitionBatch`]. The conversion process relies on trait bounds,
//! `O: From<E::Obs>` and `A: From<E::Act>`.
//!
//! # Trainer
//!
//! [`Trainer`] manages training loop and related objects. The [`Trainer`] object is
//! built with configurations of training parameters such as the maximum number of
//! optimization steps, model directory to save parameters of the agent during training, etc.
//! [`Trainer::train`] method executes online training of an agent on an environment.
//! In the training loop of this method, the agent interacts with the environment to
//! take samples and perform optimization steps. Some metrices are recorded at the same time.
//!
//! # Evaluator
//!
//! [`Evaluator<E, P>`] is used to evaluate the policy's (`P`) performance in the environment (`E`).
//! The object of this type is given to the [`Trainer`] object to evaluate the policy during training.
//! [`DefaultEvaluator<E, P>`] is a default implementation of [`Evaluator<E, P>`].
//! This evaluator runs the policy in the environment for a certain number of episodes.
//! At the start of each episode, the environment is reset using [`Env::reset_with_index()`]
//! to control specific conditions for evaluation.
//!
//! [`SimpleReplayBuffer`]: replay_buffer::SimpleReplayBuffer
//! [`SimpleReplayBuffer<O, A>`]: generic_replay_buffer::SimpleReplayBuffer
//! [`BatchBase`]: generic_replay_buffer::BatchBase
//! [`GenericTransitionBatch`]: generic_replay_buffer::GenericTransitionBatch
//! [`SimpleStepProcessor`]: replay_buffer::SimpleStepProcessor
//! [`SimpleStepProcessor<E, O, A>`]: generic_replay_buffer::SimpleStepProcessor
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

        fn save_params(&self, _path: &std::path::Path) -> anyhow::Result<()> {
            Ok(())
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
