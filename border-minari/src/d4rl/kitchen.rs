//! observation and action types, and the corresponding converter for the [Kitchen datasets](https://minari.farama.org/datasets/D4RL/kitchen/).
#[cfg(feature = "candle")]
pub mod candle;
mod evaluator;
pub mod ndarray;
pub use evaluator::KitchenEvaluator;
