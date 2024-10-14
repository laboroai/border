//! observation and action types, and the corresponding converter for the [Kitchen datasets](https://minari.farama.org/datasets/D4RL/kitchen/).
#[cfg(feature = "candle")]
pub mod candle;
pub mod ndarray;
mod evaluator;
pub use evaluator::KitchenEvaluator;
