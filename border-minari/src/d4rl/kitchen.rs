//! observation and action types, and the corresponding converter for the [Kitchen datasets](https://minari.farama.org/datasets/D4RL/kitchen/).
pub mod ndarray;
#[cfg(feature = "candle")]
pub mod candle;