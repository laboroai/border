use std::{collections::{HashMap, hash_map::Iter}, convert::Into, path::Path};
use chrono::prelude::{DateTime, Local};
use tensorboard_rs::summary_writer::SummaryWriter;

#[derive(Debug, Clone)]
/// Represents possible types of values in a [`Record`].
pub enum RecordValue {
    /// Represents a scalar, e.g., optimization steps and loss value.
    Scalar(f64), // TODO: use f32 instead of f64

    /// Represents a datetime.
    DateTime(DateTime<Local>)
}

#[derive(Debug)]
/// Represents a record.
pub struct Record (HashMap<String, RecordValue>);

impl Record {
    // pub fn new() -> Self {
    //     Self { 0: HashMap::new() }
    // }

    /// Create `Record` from slice of `(Into<String>, RecordValue)`.
    pub fn from_slice<K: Into<String> + Clone>(s: &[(K, RecordValue)]) -> Self {
        Self (s.iter().map(|(k, v)| (k.clone().into(), v.clone())).collect())
    }

    /// Insert a key-value pair into the record.
    pub fn insert(&mut self, k: impl Into<String>, v: RecordValue) {
        self.0.insert(k.into(), v);
    }

    /// Return an iterator over key-value pairs in the record.
    pub fn iter(&self) -> Iter<'_, String, RecordValue> {
        self.0.iter()
    }

    /// Get the value of the given key.
    pub fn get(&self, k: &str) -> Option<&RecordValue> {
        self.0.get(k)
    }
}

pub trait Recorder {
    fn write(&mut self, record: Record);
}

pub struct NullRecorder {}

impl NullRecorder {}

impl Recorder for NullRecorder {
    fn write(&mut self, _record: Record) {}
}

pub struct TensorboardRecorder {
    writer: SummaryWriter,
    step_key: String,
}

impl TensorboardRecorder {
    pub fn new<P: AsRef<Path>>(logdir: P) -> Self {
        Self {
            writer: SummaryWriter::new(logdir),
            step_key: "n_opts".to_string()
        }
    }
}

impl Recorder for TensorboardRecorder {
    fn write(&mut self, record: Record) {
        // TODO: handle error
        let step = match record.get(&self.step_key).unwrap() {
            RecordValue::Scalar(v) => { *v as usize },
            _ => { panic!() }
        };

        for (k, v) in record.iter() {
            if *k != self.step_key {
                match v {
                    RecordValue::Scalar(v) => {
                        self.writer.add_scalar(k, *v as f32, step)
                    },
                    RecordValue::DateTime(_) => {}, // discard value
                    // _ => { unimplemented!() }
                };                
            }
        }
    }
}
