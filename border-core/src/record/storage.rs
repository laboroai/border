use super::{Record, RecordValue};
use std::collections::HashSet;
use xxhash_rust::xxh3::Xxh3Builder;

/// Store records and aggregates them.
pub struct RecordStorage {
    data: Vec<Record>,
}

fn min(vs: &Vec<f32>) -> RecordValue {
    RecordValue::Scalar(*vs.iter().min_by(|x, y| x.total_cmp(y)).unwrap())
}

fn max(vs: &Vec<f32>) -> RecordValue {
    RecordValue::Scalar(*vs.iter().min_by(|x, y| y.total_cmp(x)).unwrap())
}

fn mean(vs: &Vec<f32>) -> RecordValue {
    RecordValue::Scalar(vs.iter().map(|v| *v).sum::<f32>() / vs.len() as f32)
}

fn median(mut vs: Vec<f32>) -> RecordValue {
    vs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    RecordValue::Scalar(vs[vs.len() / 2])
}

impl RecordStorage {
    fn get_keys(&self) -> HashSet<String, Xxh3Builder> {
        let mut keys = HashSet::<String, Xxh3Builder>::default();
        for record in self.data.iter() {
            for k in record.keys() {
                keys.insert(k.clone());
            }
        }
        keys
    }

    /// Returns a reference to a value having given key.
    fn find(&self, key: &String) -> &RecordValue {
        for record in self.data.iter() {
            if let Some(value) = record.get(key) {
                return value;
            }
        }
        panic!("Key '{}' was not found. ", key);
    }

    // Takes value from the last record.
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

    // Takes value from the last record.
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

    // Takes value from the last record.
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

    // Takes value from the last record.
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

    // Takes value from the last record.
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

    // Mean, Median, Min, Max
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

        Record::from_slice(&[
            (format!("{}_min", key), min(&vs)),
            (format!("{}_max", key), max(&vs)),
            (format!("{}_mean", key), mean(&vs)),
            (format!("{}_median", key), median(vs)),
        ])
    }

    /// Creates the storage.
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    /// Store the given record.
    pub fn store(&mut self, record: Record) {
        self.data.push(record);
    }

    /// Returns aggregated record and clear the storage.
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
            // record = record.merge(r);
            record = record.merge(r);
        }

        self.data = vec![];

        record
    }
}
