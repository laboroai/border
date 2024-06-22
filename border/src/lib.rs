//! Border is a reinforcement learning library.
//!
//! This crate is a collection of examples using the crates below.
//!
//! * [`border-core`](https://crates.io/crates/border-core) provides basic traits and functions
//!   generic to environments and reinforcmenet learning (RL) agents.
//! * [`border-py-gym-env`](https://crates.io/crates/border-py-gym-env) is a wrapper of the
//!   [Gym](https://gym.openai.com) environments written in Python, with the support of
//!   [pybullet-gym](https://github.com/benelot/pybullet-gym) and
//!   [atari](https://github.com/mgbellemare/Arcade-Learning-Environment).
//! * [`border-atari-env`](https://crates.io/crates/border-atari-env) is a wrapper of
//!   [atari-env](https://crates.io/crates/atari-env), which is a part of
//!   [gym-rs](https://crates.io/crates/gym-rs).
//! * [`border-tch-agent`](https://crates.io/crates/border-tch-agent) is a collection of RL agents
//!   based on [tch](https://crates.io/crates/tch). Deep Q network (DQN), implicit quantile network
//!   (IQN), and soft actor critic (SAC) are includes.
//! * [`border-async-trainer`](https://crates.io/crates/border-async-trainer) defines some traits and
//!   functions for asynchronous training of RL agents by multiple actors, each of which runs
//!   a sampling process of an agent and an environment in parallel.
//!
//! You can use a part of these crates for your purposes.
//!
//! # Environment
//!
//! [`border-core`](https://crates.io/crates/border-core) abstracts environments as [`Env`].
//! [`Env`] has associated types [`Env::Obs`] and [`Env::Act`] for observation and action of
//! the envirnoment. [`Env::Config`] should be configrations of the concrete type.
//!
//! # Policy and agent
//!
//! In this crate, [`Policy`] is a controller for an environment implementing [`Env`] trait.
//! [`Agent`] trait abstracts a trainable [`Policy`] and has methods for save/load of
//! parameters, and its training.
//!
//! # Evaluation
//!
//! Structs that implements [`Evaluator`] trait can be used to run episodes with a given [`Env`]
//! and [`Policy`].
//! The code might look like below. Here we use [`DefaultEvaluator`], a built-in implementation
//! of [`Evaluator`].
//!
//! ```ignore
//! type E = TYPE_OF_ENV;
//! type P = TYPE_OF_POLICY;
//!
//! fn eval(model_dir: &str, render: bool) -> Result<()> {
//!     let env_config: E::Config = {
//!         let mut env_config = env_config()
//!             .render_mode(Some("human".to_string()))
//!             .set_wait_in_millis(10);
//!         env_config
//!     };
//!     let mut agent: P = {
//!         let mut agent = create_agent();
//!         agent.load(model_dir)?;
//!         agent.eval();
//!         agent
//!     };
//!
//!     let _ = DefaultEvaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);
//! }
//! ```
//!
//! Users can customize the way the policy is evaluated by implementing a custom [`Evaluator`].
//!
//! # Training
//!
//! You can train RL [`Agent`]s by using [`Trainer`] struct.
//!
//! ```ignore
//! fn train(max_opts: usize, model_dir: &str) -> Result<()> {
//!     let mut trainer = {
//!         let env_config = env_config(); // configration of the environment
//!         let step_proc_config = SimpleStepProcessorConfig {};
//!         let replay_buffer_config =
//!             SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
//!         let config = TrainerConfig::default()
//!             .max_opts(max_opts);
//!             // followed by methods to set training parameters
//!
//!         trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
//!             config,
//!             env_config,
//!             step_proc_config,
//!             replay_buffer_config,
//!         )
//!     };
//!     let mut agent = create_agent();
//!     let mut recorder = TensorboardRecorder::new(model_dir);
//!     let mut evaluator = create_evaluator(&env_config())?;
//!
//!     trainer.train(&mut agent, &mut recorder, &mut evaluator)?;
//!
//!     Ok(())
//! }
//! ```
//! In the above code, [`SimpleStepProcessorConfig`] is configurations of
//! [`SimpleStepProcessor`], which implements [`StepProcessor`] trait.
//! [`StepProcessor`] abstracts the way how [`Step`] object is processed before pushed to
//! a replay buffer. Users can customize implementation of [`StepProcessor`] for their
//! purpose. For example, n-step TD samples or samples having Monte Carlo returns after the end
//! of episode can be computed with a statefull implementation of [`StepProcessor`].
//!
//! It should be noted that a replay buffer is not a part of [`Agent`], but owned by
//! [`Trainer`]. In the above code, the configuration of a replay buffer is given to
//! [`Trainer`]. The design choise allows [`Agent`]s to separate sampling and optimization
//! processes.
//!
//! [`border-core`]: https://crates.io/crates/border-core
//! [`Env`]: border_core::Env
//! [`Env::Obs`]: border_core::Env::Obs
//! [`Env::Act`]: border_core::Env::Act
//! [`Env::Config`]: border_core::Env::Config
//! [`Policy`]: border_core::Policy
//! [`Recorder`]: border_core::record::Recorder
//! [`eval_with_recorder`]: border_core::util::eval_with_recorder
//! [`border_py_gym_env/examples/random_cartpols.rs`]: (https://github.com/taku-y/border/blob/982ef2d25a0ade93fb71cab3bb85e5062b6f769c/border-py-gym-env/examples/random_cartpole.rs)
//! [`Agent`]: border_core::Agent
//! [`StepProcessor`]: border_core::StepProcessor
//! [`SimpleStepProcessor`]: border_core::replay_buffer::SimpleStepProcessor
//! [`SimpleStepProcessorConfig`]: border_core::replay_buffer::SimpleStepProcessorConfig
//! [`Step<E: Env>`]: border_core::Step
//! [`ReplayBufferBase`]: border_core::ReplayBufferBase
//! [`ReplayBufferBase::Batch`]: border_core::ReplayBufferBase::Batch
//! [`TransitionBatch`]: border_core::TransitionBatch
//! [`ReplayBufferBase::Batch`]: border_core::ReplayBufferBase::Batch
//! [`Agent::opt()`]: border_core::Agent::opt
//! [`ExperienceBufferBase`]: border_core::ExperienceBufferBase
//! [`ExperienceBufferBase::PushedItem`]: border_core::ExperienceBufferBase::PushedItem
//! [`SimpleReplayBuffer`]: border_core::replay_buffer::SimpleReplayBuffer
//! [`Evaluator`]: border_core::Evaluator
//! [`DefaultEvaluator`]: border_core::DefaultEvaluator
//! [`Trainer`]: border_core::Trainer
//! [`Step`]: border_core::Step

pub mod util;
