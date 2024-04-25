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

    /// Get String value.
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
}
