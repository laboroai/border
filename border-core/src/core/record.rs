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
//! in a [Recorder] given as an input argument.
//!
use chrono::prelude::{DateTime, Local};
use std::{
    collections::{
        hash_map::{IntoIter, Iter, Keys},
        HashMap,
    },
    convert::Into,
    iter::IntoIterator,
    path::Path,
};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::error::LrrError;

#[derive(Debug, Clone)]
/// Represents possible types of values in a [`Record`].
pub enum RecordValue {
    /// Represents a scalar, e.g., optimization steps and loss value.
    Scalar(f32),

    /// Represents a datetime.
    DateTime(DateTime<Local>),

    /// A 1-dimensional array
    Array1(Vec<f32>),

    /// A 2-dimensional array
    Array2(Vec<f32>, [usize; 2]),

    /// A 3-dimensional array
    Array3(Vec<f32>, [usize; 3]),
}

#[derive(Debug)]
/// Represents a record.
pub struct Record(HashMap<String, RecordValue>);

impl Record {
    /// Construct empty record.
    pub fn empty() -> Self {
        Self { 0: HashMap::new() }
    }

    /// Create `Record` from slice of `(Into<String>, RecordValue)`.
    pub fn from_slice<K: Into<String> + Clone>(s: &[(K, RecordValue)]) -> Self {
        Self(
            s.iter()
                .map(|(k, v)| (k.clone().into(), v.clone()))
                .collect(),
        )
    }

    /// Get keys.
    pub fn keys(&self) -> Keys<String, RecordValue> {
        self.0.keys()
    }

    /// Insert a key-value pair into the record.
    pub fn insert(&mut self, k: impl Into<String>, v: RecordValue) {
        self.0.insert(k.into(), v);
    }

    /// Return an iterator over key-value pairs in the record.
    pub fn iter(&self) -> Iter<'_, String, RecordValue> {
        self.0.iter()
    }

    /// Return an iterator over key-value pairs in the record.
    pub fn into_iter_in_record(self) -> IntoIter<String, RecordValue> {
        self.0.into_iter()
    }

    /// Get the value of the given key.
    pub fn get(&self, k: &str) -> Option<&RecordValue> {
        self.0.get(k)
    }

    /// Merge records.
    pub fn merge(self, record: Record) -> Self {
        Record(self.0.into_iter().chain(record.0).collect())
    }

    /// Get scalar value.
    pub fn get_scalar(&self, k: &str) -> Result<f32, LrrError> {
        if let Some(v) = self.0.get(k) {
            match v {
                RecordValue::Scalar(v) => Ok(*v as _),
                _ => Err(LrrError::RecordValueTypeError("Scalar".to_string())),
            }
        } else {
            Err(LrrError::RecordKeyError(k.to_string()))
        }
    }

    /// Get Array1 value.
    pub fn get_array1(&self, k: &str) -> Result<Vec<f32>, LrrError> {
        if let Some(v) = self.0.get(k) {
            match v {
                RecordValue::Array1(v) => Ok(v.clone()),
                _ => Err(LrrError::RecordValueTypeError("Array1".to_string())),
            }
        } else {
            Err(LrrError::RecordKeyError(k.to_string()))
        }
    }

    /// Get Array2 value.
    pub fn get_array2(&self, k: &str) -> Result<(Vec<f32>, [usize; 2]), LrrError> {
        if let Some(v) = self.0.get(k) {
            match v {
                RecordValue::Array2(v, s) => Ok((v.clone(), s.clone())),
                _ => Err(LrrError::RecordValueTypeError("Array2".to_string())),
            }
        } else {
            Err(LrrError::RecordKeyError(k.to_string()))
        }
    }

    /// Get Array3 value.
    pub fn get_array3(&self, k: &str) -> Result<(Vec<f32>, [usize; 3]), LrrError> {
        if let Some(v) = self.0.get(k) {
            match v {
                RecordValue::Array3(v, s) => Ok((v.clone(), s.clone())),
                _ => Err(LrrError::RecordValueTypeError("Array3".to_string())),
            }
        } else {
            Err(LrrError::RecordKeyError(k.to_string()))
        }
    }
}

/// Process records provided with [`Recorder::write`]
pub trait Recorder {
    /// Write a record to the [`Recorder`].
    fn write(&mut self, record: Record);
}

/// A recorder that ignores any record. This struct is used just for debugging.
pub struct NullRecorder {}

impl NullRecorder {}

impl Recorder for NullRecorder {
    /// Discard the given record.
    fn write(&mut self, _record: Record) {}
}

/// Write records to TFRecord.
pub struct TensorboardRecorder {
    writer: SummaryWriter,
    step_key: String,
}

impl TensorboardRecorder {
    /// Construct a [`TensorboardRecorder`].
    ///
    /// TFRecord will be stored in `logdir`.
    pub fn new<P: AsRef<Path>>(logdir: P) -> Self {
        Self {
            writer: SummaryWriter::new(logdir),
            step_key: "n_opts".to_string(),
        }
    }
}

impl Recorder for TensorboardRecorder {
    /// Write a given [Record] into a TFRecord.
    ///
    /// This method handles [RecordValue::Scalar] and [RecordValue::DateTime] in the [Record].
    /// Other variants will be ignored.
    fn write(&mut self, record: Record) {
        // TODO: handle error
        let step = match record.get(&self.step_key).unwrap() {
            RecordValue::Scalar(v) => *v as usize,
            _ => {
                panic!()
            }
        };

        for (k, v) in record.iter() {
            if *k != self.step_key {
                match v {
                    RecordValue::Scalar(v) => self.writer.add_scalar(k, *v as f32, step),
                    RecordValue::DateTime(_) => {} // discard value
                    _ => {
                        unimplemented!()
                    }
                };
            }
        }
    }
}

/// Buffered recorder.
///
/// This is used for recording sequences of observation and action
/// during evaluation runs in [`crate::core::util::eval_with_recorder`].
#[derive(Default)]
pub struct BufferedRecorder(Vec<Record>);

impl BufferedRecorder {
    /// Construct the recorder.
    pub fn new() -> Self {
        Self(Vec::default())
    }

    /// Returns an iterator over the records.
    pub fn iter(&self) -> std::slice::Iter<Record> {
        self.0.iter()
    }
}

impl Recorder for BufferedRecorder {
    /// Write a [`Record`] to the buffer.
    ///
    /// TODO: Consider if it is worth making the method public.
    fn write(&mut self, record: Record) {
        self.0.push(record);
    }
}
