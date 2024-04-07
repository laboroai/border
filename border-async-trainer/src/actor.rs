//! Takes samples from the environment and pushes them to the replay buffer.
mod base;
mod stat;
pub use base::Actor;
pub use stat::{actor_stats_fmt, ActorStat};
