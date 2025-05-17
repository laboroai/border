//! Record storage and aggregation implementation.
//!
//! This module provides functionality for storing and aggregating records
//! during training and evaluation. It supports various types of data
//! aggregation, including statistical measures for scalar values and
//! handling of different data types.

use super::{Record, RecordValue};
use std::collections::HashSet;
use xxhash_rust::xxh3::Xxh3Builder;

/// A storage system for records with aggregation capabilities.
///
/// `RecordStorage` provides functionality to store records and perform
/// various types of aggregation on the stored data. It supports:
///
/// * Storing multiple records
/// * Aggregating scalar values with statistical measures (min, max, mean, median)
/// * Handling different data types (scalars, arrays, strings, datetimes)
/// * Automatic type-based aggregation
pub struct RecordStorage {
    /// Internal storage for records
    data: Vec<Record>,
}

/// Calculates the minimum value from a vector of floats.
///
/// # Arguments
///
/// * `vs` - A vector of floating-point values
///
/// # Returns
///
/// A `RecordValue::Scalar` containing the minimum value
fn min(vs: &Vec<f32>) -> RecordValue {
    RecordValue::Scalar(*vs.iter().min_by(|x, y| x.total_cmp(y)).unwrap())
}

/// Calculates the maximum value from a vector of floats.
///
/// # Arguments
///
/// * `vs` - A vector of floating-point values
///
/// # Returns
///
/// A `RecordValue::Scalar` containing the maximum value
fn max(vs: &Vec<f32>) -> RecordValue {
    RecordValue::Scalar(*vs.iter().min_by(|x, y| y.total_cmp(x)).unwrap())
}

/// Calculates the mean value from a vector of floats.
///
/// # Arguments
///
/// * `vs` - A vector of floating-point values
///
/// # Returns
///
/// A `RecordValue::Scalar` containing the mean value
fn mean(vs: &Vec<f32>) -> RecordValue {
    RecordValue::Scalar(vs.iter().map(|v| *v).sum::<f32>() / vs.len() as f32)
}

/// Calculates the median value from a vector of floats.
///
/// # Arguments
///
/// * `vs` - A vector of floating-point values
///
/// # Returns
///
/// A `RecordValue::Scalar` containing the median value
///
/// # Note
///
/// This function sorts the input vector in place.
fn median(mut vs: Vec<f32>) -> RecordValue {
    vs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    RecordValue::Scalar(vs[vs.len() / 2])
}

impl RecordStorage {
    /// Returns a set of all unique keys in the stored records.
    ///
    /// # Returns
    ///
    /// A `HashSet` containing all unique keys from the stored records
    fn get_keys(&self) -> HashSet<String, Xxh3Builder> {
        let mut keys = HashSet::<String, Xxh3Builder>::default();
        for record in self.data.iter() {
            for k in record.keys() {
                keys.insert(k.clone());
            }
        }
        keys
    }

    /// Finds the first occurrence of a value with the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// A reference to the first matching value
    ///
    /// # Panics
    ///
    /// Panics if the key is not found in any record
    fn find(&self, key: &String) -> &RecordValue {
        for record in self.data.iter() {
            if let Some(value) = record.get(key) {
                return value;
            }
        }
        panic!("Key '{}' was not found. ", key);
    }

    /// Gets the most recent datetime value for a given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the datetime value
    ///
    /// # Returns
    ///
    /// A new record containing the datetime value
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The key is not found
    /// - The value is not a datetime
    fn datetime(&self, key: &String) -> Record {
        for record in self.data.iter().rev() {
            if let Some(value) = record.get(key) {
                match value {
                    RecordValue::DateTime(..) => {
                        return Record::from_slice(&[(key, value.clone())]);
                    }
                    _ => panic!("Expect RecordValue::DateTime for {}", key),
                }
            }
        }
        panic!("Unexpected");
    }

    /// Gets the most recent 1D array value for a given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the array value
    ///
    /// # Returns
    ///
    /// A new record containing the array value
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The key is not found
    /// - The value is not a 1D array
    fn array1(&self, key: &String) -> Record {
        for record in self.data.iter().rev() {
            if let Some(value) = record.get(key) {
                match value {
                    RecordValue::Array1(..) => {
                        return Record::from_slice(&[(key, value.clone())]);
                    }
                    _ => panic!("Expect RecordValue::Array1 for {}", key),
                }
            }
        }
        panic!("Unexpected");
    }

    /// Gets the most recent 2D array value for a given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the array value
    ///
    /// # Returns
    ///
    /// A new record containing the array value
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The key is not found
    /// - The value is not a 2D array
    fn array2(&self, key: &String) -> Record {
        for record in self.data.iter().rev() {
            if let Some(value) = record.get(key) {
                match value {
                    RecordValue::Array2(..) => {
                        return Record::from_slice(&[(key, value.clone())]);
                    }
                    _ => panic!("Expect RecordValue::Array2 for {}", key),
                }
            }
        }
        panic!("Unexpected");
    }

    /// Gets the most recent 3D array value for a given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the array value
    ///
    /// # Returns
    ///
    /// A new record containing the array value
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The key is not found
    /// - The value is not a 3D array
    fn array3(&self, key: &String) -> Record {
        for record in self.data.iter().rev() {
            if let Some(value) = record.get(key) {
                match value {
                    RecordValue::Array3(..) => {
                        return Record::from_slice(&[(key, value.clone())]);
                    }
                    _ => panic!("Expect RecordValue::Array3 for {}", key),
                }
            }
        }
        panic!("Unexpected");
    }

    /// Gets the most recent string value for a given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the string value
    ///
    /// # Returns
    ///
    /// A new record containing the string value
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The key is not found
    /// - The value is not a string
    fn string(&self, key: &String) -> Record {
        for record in self.data.iter().rev() {
            if let Some(value) = record.get(key) {
                match value {
                    RecordValue::String(..) => {
                        return Record::from_slice(&[(key, value.clone())]);
                    }
                    _ => panic!("Expect RecordValue::String for {}", key),
                }
            }
        }
        panic!("Unexpected");
    }

    /// Aggregates scalar values with statistical measures.
    ///
    /// For a single value, returns it directly. For multiple values,
    /// calculates min, max, mean, and median.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the scalar values
    ///
    /// # Returns
    ///
    /// A new record containing the aggregated values
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The key is not found
    /// - The values are not scalars
    fn scalar(&self, key: &String) -> Record {
        let vs: Vec<f32> = self
            .data
            .iter()
            .filter_map(|record| match record.get(key) {
                Some(v) => match v {
                    RecordValue::Scalar(v) => Some(*v),
                    _ => panic!("Expect RecordValue::Scalar for {}", key),
                },
                None => None,
            })
            .collect();

        if vs.len() == 1 {
            Record::from_slice(&[(format!("{}", key), RecordValue::Scalar(vs[0]))])
        } else {
            Record::from_slice(&[
                (format!("{}_min", key), min(&vs)),
                (format!("{}_max", key), max(&vs)),
                (format!("{}_mean", key), mean(&vs)),
                (format!("{}_median", key), median(vs)),
            ])
        }
    }

    /// Creates a new empty record storage.
    ///
    /// # Returns
    ///
    /// A new empty `RecordStorage` instance
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    /// Stores a record in the storage.
    ///
    /// # Arguments
    ///
    /// * `record` - The record to store
    pub fn store(&mut self, record: Record) {
        self.data.push(record);
    }

    /// Aggregates all stored records and clears the storage.
    ///
    /// This method:
    /// 1. Collects all unique keys from stored records
    /// 2. Aggregates values based on their type
    /// 3. Creates a new record with aggregated values
    /// 4. Clears the storage
    ///
    /// # Returns
    ///
    /// A new record containing all aggregated values
    pub fn aggregate(&mut self) -> Record {
        let mut record = Record::empty();

        for key in self.get_keys().iter() {
            let value = self.find(key);
            let r = match value {
                RecordValue::DateTime(..) => self.datetime(key),
                RecordValue::Array1(..) => self.array1(key),
                RecordValue::Array2(..) => self.array2(key),
                RecordValue::Array3(..) => self.array3(key),
                RecordValue::String(..) => self.string(key),
                RecordValue::Scalar(..) => self.scalar(key),
            };
            record = record.merge(r);
        }

        self.data = vec![];

        record
    }
}
