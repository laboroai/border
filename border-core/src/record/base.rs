//! Base implementation of records for logging.
//!
//! This module provides a flexible record system for storing and retrieving
//! various types of data, including scalars, arrays, and strings. It is designed
//! to be used for logging training metrics, storing experiment results, and
//! managing configuration data.

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

/// Represents possible types of values that can be stored in a [`Record`].
///
/// This enum provides a type-safe way to store different kinds of data in a record,
/// including numerical values, arrays, strings, and timestamps.
///
/// # Variants
///
/// * `Scalar(f32)` - A single floating-point value, typically used for metrics
/// * `DateTime(DateTime<Local>)` - A timestamp with local timezone
/// * `Array1(Vec<f32>)` - A 1-dimensional array of floating-point values
/// * `Array2(Vec<f32>, [usize; 2])` - A 2-dimensional array with shape information
/// * `Array3(Vec<f32>, [usize; 3])` - A 3-dimensional array with shape information
/// * `String(String)` - A text value
#[derive(Debug, Clone)]
pub enum RecordValue {
    /// A single floating-point value, typically used for metrics like loss or accuracy.
    Scalar(f32),

    /// A timestamp with local timezone, useful for logging events.
    DateTime(DateTime<Local>),

    /// A 1-dimensional array of floating-point values.
    Array1(Vec<f32>),

    /// A 2-dimensional array with shape information.
    Array2(Vec<f32>, [usize; 2]),

    /// A 3-dimensional array with shape information.
    Array3(Vec<f32>, [usize; 3]),

    /// A text value, useful for storing labels or descriptions.
    String(String),
}

/// A container for storing key-value pairs of various data types.
///
/// This structure provides a flexible way to store and retrieve different types
/// of data using string keys. It supports merging records and provides type-safe
/// access to stored values.
///
/// # Examples
///
/// ```rust
/// use border_core::record::{Record, RecordValue};
///
/// // Create a record with a scalar value
/// let mut record = Record::from_scalar("loss", 0.5);
///
/// // Add more values
/// record.insert("accuracy", RecordValue::Scalar(0.95));
/// record.insert("timestamp", RecordValue::DateTime(Local::now()));
///
/// // Retrieve values
/// let loss = record.get_scalar("loss").unwrap();
/// ```
#[derive(Debug)]
pub struct Record(HashMap<String, RecordValue>);

impl Record {
    /// Creates an empty record.
    ///
    /// # Returns
    ///
    /// A new empty record
    pub fn empty() -> Self {
        Self { 0: HashMap::new() }
    }

    /// Creates a record containing a single scalar value.
    ///
    /// # Arguments
    ///
    /// * `name` - The key for the scalar value
    /// * `value` - The scalar value to store
    ///
    /// # Returns
    ///
    /// A new record containing the scalar value
    pub fn from_scalar(name: impl Into<String>, value: f32) -> Self {
        Self {
            0: HashMap::from([(name.into(), RecordValue::Scalar(value))]),
        }
    }

    /// Creates a record from a slice of key-value pairs.
    ///
    /// # Arguments
    ///
    /// * `s` - A slice of tuples containing keys and values
    ///
    /// # Returns
    ///
    /// A new record containing all the key-value pairs
    pub fn from_slice<K: Into<String> + Clone>(s: &[(K, RecordValue)]) -> Self {
        Self(
            s.iter()
                .map(|(k, v)| (k.clone().into(), v.clone()))
                .collect(),
        )
    }

    /// Returns an iterator over the keys in the record.
    ///
    /// # Returns
    ///
    /// An iterator over the record's keys
    pub fn keys(&self) -> Keys<String, RecordValue> {
        self.0.keys()
    }

    /// Inserts a key-value pair into the record.
    ///
    /// # Arguments
    ///
    /// * `k` - The key to insert
    /// * `v` - The value to insert
    pub fn insert(&mut self, k: impl Into<String>, v: RecordValue) {
        self.0.insert(k.into(), v);
    }

    /// Returns an iterator over the key-value pairs in the record.
    ///
    /// # Returns
    ///
    /// An iterator over the record's key-value pairs
    pub fn iter(&self) -> Iter<'_, String, RecordValue> {
        self.0.iter()
    }

    /// Returns an iterator that consumes the record.
    ///
    /// # Returns
    ///
    /// An iterator that takes ownership of the record
    pub fn into_iter_in_record(self) -> IntoIter<String, RecordValue> {
        self.0.into_iter()
    }

    /// Gets a reference to the value associated with the given key.
    ///
    /// # Arguments
    ///
    /// * `k` - The key to look up
    ///
    /// # Returns
    ///
    /// A reference to the value if the key exists, `None` otherwise
    pub fn get(&self, k: &str) -> Option<&RecordValue> {
        self.0.get(k)
    }

    /// Merges two records, consuming both.
    ///
    /// If both records contain the same key, the value from the second record
    /// will overwrite the value from the first record.
    ///
    /// # Arguments
    ///
    /// * `record` - The record to merge with
    ///
    /// # Returns
    ///
    /// A new record containing all key-value pairs from both records
    pub fn merge(self, record: Record) -> Self {
        Record(self.0.into_iter().chain(record.0).collect())
    }

    /// Merges another record into this one in place.
    ///
    /// If both records contain the same key, the value from the second record
    /// will overwrite the value from this record.
    ///
    /// # Arguments
    ///
    /// * `record` - The record to merge with
    pub fn merge_inplace(&mut self, record: Record) {
        for (k, v) in record.iter() {
            self.0.insert(k.clone(), v.clone());
        }
    }

    /// Gets a scalar value from the record.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the scalar value
    ///
    /// # Returns
    ///
    /// The scalar value if it exists and is of the correct type
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The key does not exist
    /// - The value is not a scalar
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

    /// Gets a 1-dimensional array from the record.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the array
    ///
    /// # Returns
    ///
    /// The array if it exists and is of the correct type
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The key does not exist
    /// - The value is not a 1-dimensional array
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

    /// Gets a 2-dimensional array from the record.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the array
    ///
    /// # Returns
    ///
    /// A tuple containing the array data and its shape
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The key does not exist
    /// - The value is not a 2-dimensional array
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

    /// Gets a 3-dimensional array from the record.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the array
    ///
    /// # Returns
    ///
    /// A tuple containing the array data and its shape
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The key does not exist
    /// - The value is not a 3-dimensional array
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

    /// Gets a string value from the record.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the string
    ///
    /// # Returns
    ///
    /// The string if it exists and is of the correct type
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The key does not exist
    /// - The value is not a string
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

    /// Checks if the record is empty.
    ///
    /// # Returns
    ///
    /// `true` if the record contains no key-value pairs
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// Gets a scalar value from the record without specifying a key.
    ///
    /// This method is useful when the record contains only one scalar value.
    ///
    /// # Returns
    ///
    /// The scalar value if it exists and is the only value in the record
    pub fn get_scalar_without_key(&self) -> Option<f32> {
        if self.0.len() != 1 {
            return None;
        } else {
            let key = self.0.keys().next().unwrap();
            match self.0.get(key) {
                Some(RecordValue::Scalar(value)) => Some(*value),
                _ => None,
            }
        }
    }
}
