//! Observation and action types, and the corresponding converter for the
//! [Point Maze datasets](https://minari.farama.org/datasets/D4RL/pointmaze/).
#[cfg(feature = "candle")]
pub mod candle;
mod evaluator;
pub mod ndarray;
pub use evaluator::PointMazeEvaluator;
