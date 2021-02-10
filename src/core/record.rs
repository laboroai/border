use std::{collections::HashMap, convert::Into};
use chrono::prelude::{DateTime, Local};

#[derive(Debug, Clone)]
pub enum RecordValue {
    Scalar(f64),
    DateTime(DateTime<Local>)
}

pub struct Record (HashMap<String, RecordValue>);

impl Record {
    // pub fn new() -> Self {
    //     Self { 0: HashMap::new() }
    // }

    /// Create `Record` from slice of `(Into<String>, RecordValue)`.
    pub fn from_slice<K: Into<String> + Clone>(s: &[(K, RecordValue)]) -> Self {
        Self (s.iter().map(|(k, v)| (k.clone().into(), v.clone())).collect())
    }

    pub fn insert(&mut self, k: impl Into<String>, v: RecordValue) {
        self.0.insert(k.into(), v);
    }
}

pub trait TrainRecorder {
    fn write(&mut self, record: Record);
}

pub struct NullTrainRecorder {}

impl NullTrainRecorder {
    pub fn new() -> Self { Self {} }
}

impl TrainRecorder for NullTrainRecorder {
    fn write(&mut self, _record: Record) {}
}
