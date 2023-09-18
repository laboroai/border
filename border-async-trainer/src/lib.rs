//! Asynchronous trainer with parallel sampling processes.
//!
//! The code might look like below.
//!
//! ```ignore
//! fn train() {
//!     let agent_configs: Vec<_> = vec![agent_config()];
//!     let env_config_train = env_config(name);
//!     let env_config_eval = env_config(name).eval();
//!     let replay_buffer_config = load_replay_buffer_config(model_dir.as_str())?;
//!     let step_proc_config = SimpleStepProcessorConfig::default();
//!     let actor_man_config = ActorManagerConfig::default();
//!     let async_trainer_config = load_async_trainer_config(model_dir.as_str())?;
//!     let mut recorder = TensorboardRecorder::new(model_dir);
//!     let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;
//!
//!     // Shared flag to stop actor threads
//!     let stop = Arc::new(Mutex::new(false));
//!
//!     // Creates channels
//!     let (item_s, item_r) = unbounded(); // items pushed to replay buffer
//!     let (model_s, model_r) = unbounded(); // model_info
//!
//!     // guard for initialization of envs in multiple threads
//!     let guard_init_env = Arc::new(Mutex::new(true));
//!
//!     // Actor manager and async trainer
//!     let mut actors = ActorManager::build(
//!         &actor_man_config,
//!         &agent_configs,
//!         &env_config_train,
//!         &step_proc_config,
//!         item_s,
//!         model_r,
//!         stop.clone(),
//!     );
//!     let mut trainer = AsyncTrainer::build(
//!         &async_trainer_config,
//!         &agent_config,
//!         &env_config_eval,
//!         &replay_buffer_config,
//!         item_r,
//!         model_s,
//!         stop.clone(),
//!     );
//!
//!     // Set the number of threads
//!     tch::set_num_threads(1);
//!
//!     // Starts sampling and training
//!     actors.run(guard_init_env.clone());
//!     let stats = trainer.train(&mut recorder, &mut evaluator, guard_init_env);
//!     println!("Stats of async trainer");
//!     println!("{}", stats.fmt());
//!
//!     let stats = actors.stop_and_join();
//!     println!("Stats of generated samples in actors");
//!     println!("{}", actor_stats_fmt(&stats));
//! }
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
