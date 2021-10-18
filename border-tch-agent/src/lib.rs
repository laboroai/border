//! RL agents implemented with [tch](https://crates.io/crates/tch).
pub mod cnn;
pub mod dqn;
// pub mod iqn;
pub mod mlp;
pub mod model;
pub mod opt;
pub mod sac;
mod tensor_batch;
// pub mod replay_buffer;
pub mod util;
pub use tensor_batch::{TensorSubBatch, ZeroTensor};
