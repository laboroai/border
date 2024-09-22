//! RL agents implemented with [candle](https://crates.io/crates/candle-core).
pub mod cnn;
pub mod dqn;
// pub mod iqn;
pub mod bc;
pub mod mlp;
pub mod model;
pub mod opt;
pub mod sac;
mod tensor_batch;
pub mod util;
use candle_core::{backend::BackendDevice, DeviceLocation};
use serde::{Deserialize, Serialize};
pub use tensor_batch::{TensorBatch, ZeroTensor};

#[derive(Clone, Debug, Copy, Deserialize, Serialize, PartialEq)]
/// Device for using candle.
///
/// This enum is added because [`candle_core::Device`] does not support serialization.
///
/// [`candle_core::Device`]: https://docs.rs/candle-core/0.4.1/candle_core/enum.Device.html
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
            candle_core::Device::Cuda(cuda_device) => {
                let loc = cuda_device.location();
                match loc {
                    DeviceLocation::Cuda { gpu_id } => Self::Cuda(gpu_id),
                    _ => panic!(),
                }
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
