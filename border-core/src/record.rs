//! Types for recording various values obtained during training and evaluation.
//!
//! [Record] is a [HashMap], where its key and values represents various values obtained during training and
//! evaluation. A record may contains multiple types of values.
//!
//! ```no_run
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
//! A typical usecase is to record internal values obtained in training processes.
//! [Trainer::train](crate::Trainer::train), which executes a training loop, writes a record
//! in a [`Recorder`] given as an input argument.
//!
mod base;
mod buffered_recorder;
mod null_recorder;
mod recorder;

pub use base::{Record, RecordValue};
pub use buffered_recorder::BufferedRecorder;
pub use null_recorder::NullRecorder;
pub use recorder::Recorder;
