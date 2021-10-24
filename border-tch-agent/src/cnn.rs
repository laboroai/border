//! Convolutional neural network.
//!
//! The architecture is the same in the DQN Nature paper.
//! It should be noted that the input array will be scaled by 1 / 255 for normalizing
//! pixel intensities with casting from `u8` to `f32`.
mod base;
mod config;
pub use base::CNN;
pub use config::CNNConfig;
