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
use serde::{Deserialize, Serialize};
pub use tensor_batch::{TensorSubBatch, ZeroTensor};

#[derive(Clone, Debug, Copy, Deserialize, Serialize, PartialEq)]
/// Device for using tch-rs.
///
/// This enum is added because `tch::Device` does not support serialization.
pub enum Device {
    /// The main CPU device.
    Cpu,

    /// The main GPU device.
    Cuda(usize),
}

impl From<tch::Device> for Device {
    fn from(device: tch::Device) -> Self {
        match device {
            tch::Device::Cpu => Self::Cpu,
            tch::Device::Cuda(n) => Self::Cuda(n),
        }
    }
}

impl Into<tch::Device> for Device {
    fn into(self) -> tch::Device {
        match self {
            Self::Cpu => tch::Device::Cpu,
            Self::Cuda(n) => tch::Device::Cuda(n),
        }
    }
}
