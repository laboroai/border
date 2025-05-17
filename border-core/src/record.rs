//! Types and traits for recording and managing training metrics.
//!
//! This module provides a comprehensive system for recording, storing, and managing
//! various types of data during reinforcement learning training and evaluation.
//! It includes support for different data types, aggregation, and flexible storage
//! backends.
//!
//! # Core Components
//!
//! * [`Record`] - A flexible container for storing key-value pairs of various data types
//! * [`RecordValue`] - An enum representing different types of values that can be stored
//! * [`Recorder`] - A trait defining the interface for recording and storing data
//! * [`RecordStorage`] - A storage system with aggregation capabilities
//! * [`BufferedRecorder`] - A recorder that temporarily stores records in memory
//! * [`NullRecorder`] - A recorder that discards all records (useful for testing)
//!
//! # Basic Usage
//!
//! ```rust
//! use border_core::record::{Record, RecordValue};
//!
//! // following values are obtained with some process in reality
//! let step = 1;
//! let obs = vec![1f32, 2.0, 3.0, 4.0, 5.0];
//! let reward = -1f32;
//!
//! let mut record = Record::empty();
//! record.insert("Step", RecordValue::Scalar(step as f32));
//! record.insert("Reward", RecordValue::Scalar(reward));
//! record.insert("Obs", RecordValue::Array1(obs));
//! ```
//!
//! # Integration with Training
//!
//! The recording system is designed to work seamlessly with the training process.
//! The [`Trainer`](crate::Trainer) uses a [`Recorder`] to log training metrics
//! and other relevant information during the training loop.
//!
//! # Data Types
//!
//! The module supports various data types through [`RecordValue`]:
//!
//! * `Scalar(f32)` - Single floating-point values
//! * `DateTime(DateTime<Local>)` - Timestamps
//! * `Array1(Vec<f32>)` - 1-dimensional arrays
//! * `Array2(Vec<f32>, [usize; 2])` - 2-dimensional arrays with shape
//! * `Array3(Vec<f32>, [usize; 3])` - 3-dimensional arrays with shape
//! * `String(String)` - Text values
//!
//! [`HashMap`]: std::collections::HashMap
mod base;
mod buffered_recorder;
mod null_recorder;
mod recorder;
mod storage;

pub use base::{Record, RecordValue};
pub use buffered_recorder::BufferedRecorder;
pub use null_recorder::NullRecorder;
pub use recorder::Recorder;
pub use storage::RecordStorage;
