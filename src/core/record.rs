use std::{collections::{HashMap, hash_map::Iter}, convert::Into, path::Path};
use chrono::prelude::{DateTime, Local};
use tensorboard_rs::summary_writer::SummaryWriter;

#[derive(Debug, Clone)]
pub enum RecordValue {
    Scalar(f64), // TODO: use f32 instead of f64
    DateTime(DateTime<Local>)
}

#[derive(Debug)]
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

    pub fn iter(&self) -> Iter<'_, String, RecordValue> {
        self.0.iter()
    }

    pub fn get(&self, k: &String) -> Option<&RecordValue> {
        self.0.get(k)
    }
}

pub trait TrainRecorder {
    fn write(&mut self, record: Record);
}

pub struct NullTrainRecorder {}

impl NullTrainRecorder {}

impl TrainRecorder for NullTrainRecorder {
    fn write(&mut self, _record: Record) {}
}

pub struct TensorboardTrainRecorder {
    writer: SummaryWriter,
    step_key: String,
}

impl TensorboardTrainRecorder {
    pub fn new<P: AsRef<Path>>(logdir: P) -> Self {
        Self {
            writer: SummaryWriter::new(logdir),
            step_key: "n_opts".to_string()
        }
    }
}

impl TrainRecorder for TensorboardTrainRecorder {
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
