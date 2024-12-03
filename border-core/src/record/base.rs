use crate::error::LrrError;
use chrono::prelude::{DateTime, Local};
use std::{
    collections::{
        hash_map::{IntoIter, Iter, Keys},
        HashMap,
    },
    convert::Into,
    iter::IntoIterator,
};

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

    /// String
    String(String),
}

#[derive(Debug)]
/// Represents a record.
pub struct Record(HashMap<String, RecordValue>);

impl Record {
    /// Creates empty record.
    pub fn empty() -> Self {
        Self { 0: HashMap::new() }
    }

    /// Creates a record from a scalar.
    pub fn from_scalar(name: impl Into<String>, value: f32) -> Self {
        Self {
            0: HashMap::from([(name.into(), RecordValue::Scalar(value))]),
        }
    }

    /// Creates `Record` from slice of `(Into<String>, RecordValue)`.
    pub fn from_slice<K: Into<String> + Clone>(s: &[(K, RecordValue)]) -> Self {
        Self(
            s.iter()
                .map(|(k, v)| (k.clone().into(), v.clone()))
                .collect(),
        )
    }

    /// Gets keys.
    pub fn keys(&self) -> Keys<String, RecordValue> {
        self.0.keys()
    }

    /// Inserts a key-value pair into the record.
    pub fn insert(&mut self, k: impl Into<String>, v: RecordValue) {
        self.0.insert(k.into(), v);
    }

    /// Returns an iterator over key-value pairs in the record.
    pub fn iter(&self) -> Iter<'_, String, RecordValue> {
        self.0.iter()
    }

    /// Returns an iterator over key-value pairs in the record.
    pub fn into_iter_in_record(self) -> IntoIter<String, RecordValue> {
        self.0.into_iter()
    }

    /// Gets the value of the given key.
    pub fn get(&self, k: &str) -> Option<&RecordValue> {
        self.0.get(k)
    }

    /// Merges records.
    pub fn merge(self, record: Record) -> Self {
        Record(self.0.into_iter().chain(record.0).collect())
    }

    /// Merges records.
    pub fn merge_inplace(&mut self, record: Record) {
        for (k, v) in record.iter() {
            self.0.insert(k.clone(), v.clone());
        }
    }

    /// Gets scalar value.
    ///
    /// * `key` - The key of an entry in the record.
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

    /// Gets Array1 value.
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

    /// Gets Array2 value.
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

    /// Gets Array3 value.
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

    /// Gets String value.
    pub fn get_string(&self, k: &str) -> Result<String, LrrError> {
        if let Some(v) = self.0.get(k) {
            match v {
                RecordValue::String(s) => Ok(s.clone()),
                _ => Err(LrrError::RecordValueTypeError("String".to_string())),
            }
        } else {
            Err(LrrError::RecordKeyError(k.to_string()))
        }
    }

    /// Returns true if the record is empty.
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// Gets a scalar value without specifying a key.
    /// 
    /// It works if the record has a single element of a [`RecordValue::Scalar`].
    /// Otherwise returns `None`.
    pub fn get_scalar_without_key(&self) -> Option<f32> {
        if self.0.len() != 1 {
            return None
        } else {
            let key = self.0.keys().next().unwrap();
            match self.0.get(key) {
                Some(RecordValue::Scalar(value)) => Some(*value),
                _ => None,
            }
        }
    }
}
