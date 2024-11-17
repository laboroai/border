//! observation and action types, and the corresponding converter for the
//! [Ant Maze datasets](https://minari.farama.org/datasets/D4RL/antmaze/).
// #[cfg(feature = "candle")]
// pub mod candle;
mod evaluator;
pub mod ndarray;
pub use evaluator::AntMazeEvaluator;
