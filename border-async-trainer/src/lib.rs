//! Asynchronous trainer using replay buffer.
//!
//! Training process consists of the following two components:
//! * [ActorManager] manages [Actor]s, each of which runs a thread for interacting 
//!   `Agent` and `Env` and taking samples. Those samples will be sent to
//!   the replay buffer in [AsyncTrainer].
//! * [AsyncTrainer] is responsible for training of an agent. It also runs a thread
//!   for pushing samples from [ActorManager] into a replay buffer.
//!
//! The `Agent` must implement [SyncModel] trait in order to synchronize the model of
//! the agent in [Actor] with the trained agent in [AsyncTrainer]. The trait has
//! the ability to import and export the information of the model as
//! [SyncModel]`::ModelInfo`.
//!
//! The `Agent` in [AsyncTrainer] is responsible for training, typically with a GPU,
//! while the `Agent`s in [Actor]s in [ActorManager] is responsible for sampling
//! using CPU.
//!
//! Both [AsyncTrainer] and [ActorManager] are running in the same machine and
//! communicate by channels.
mod actor;
mod actor_manager;
mod async_trainer;
mod messages;
mod replay_buffer_proxy;
mod sync_model;
mod error;
pub use actor::{Actor, ActorStat, actor_stats_fmt};
pub use actor_manager::{ActorManager, ActorManagerConfig};
pub use async_trainer::{AsyncTrainer, AsyncTrainerConfig, AsyncTrainStat};
pub use messages::PushedItemMessage;
pub use replay_buffer_proxy::{ReplayBufferProxy, ReplayBufferProxyConfig};
pub use sync_model::SyncModel;
pub use error::BorderAsyncTrainerError;