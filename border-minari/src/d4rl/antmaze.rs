//! observation and action types, and the corresponding converter for the
//! [Ant Maze datasets](https://minari.farama.org/datasets/D4RL/antmaze/).
#[cfg(feature = "candle")]
pub mod candle;
pub mod ndarray;
