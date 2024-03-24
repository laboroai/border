//! RL agents implemented with [candle](https://crates.io/crates/candle-core).
pub mod cnn;
pub mod dqn;
// pub mod iqn;
pub mod mlp;
pub mod model;
pub mod opt;
pub mod sac;
mod tensor_batch;
pub mod util;
use serde::{Deserialize, Serialize};
pub use tensor_batch::{TensorSubBatch, ZeroTensor};

#[derive(Clone, Debug, Copy, Deserialize, Serialize, PartialEq)]
/// Device for using candle.
///
/// This enum is added because [`candle_core::Device`] does not support serialization.
pub enum Device {
    /// The main CPU device.
    Cpu,

    /// The main GPU device.
    Cuda(usize),
}

impl From<candle_core::Device> for Device {
    fn from(device: candle_core::Device) -> Self {
        match device {
            candle_core::Device::Cpu => Self::Cpu,
            candle_core::Device::Cuda(_cuda_device) => {
                unimplemented!();
                // let n = cuda_device.
                // Self::Cuda(n)
            }
            _ => unimplemented!(),
        }
    }
}

impl Into<candle_core::Device> for Device {
    fn into(self) -> candle_core::Device {
        match self {
            Self::Cpu => candle_core::Device::Cpu,
            Self::Cuda(n) => candle_core::Device::new_cuda(n).unwrap(),
        }
    }
}
