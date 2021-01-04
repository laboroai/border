pub mod tch;
pub use self::tch::replay_buffer::{ReplayBuffer, TchBufferableActInfo, TchBufferableObsInfo};
pub use self::tch::dqn::{DQN, QNetwork};
pub use self::tch::Model;
