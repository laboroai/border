//! Null recorder implementation for discarding record data.
//!
//! This module provides a null recorder that discards all records without storing them.
//! It is particularly useful for debugging, testing, or when record storage is not needed
//! but the recorder interface must be maintained.

use std::marker::PhantomData;

use super::{Record, Recorder};
use crate::{Env, ReplayBufferBase};

/// A recorder that discards all records without storing them.
///
/// The `NullRecorder` implements the [`Recorder`] trait but ignores all records
/// passed to it. This is useful in several scenarios:
///
/// * During debugging when you want to disable logging temporarily
/// * In testing environments where record storage is not needed
/// * When you want to measure performance without the overhead of record storage
/// * As a placeholder when record storage is optional
///
/// # Type Parameters
///
/// * `E` - The environment type that implements the [`Env`] trait
/// * `R` - The replay buffer type that implements the [`ReplayBufferBase`] trait
///
/// # Examples
///
/// ```rust
/// use border_core::{
///     record::{NullRecorder, Record, Recorder},
///     Env, ReplayBufferBase,
/// };
///
/// // Create a new null recorder
/// let mut recorder = NullRecorder::<MyEnv, MyReplayBuffer>::new();
///
/// // All records are silently discarded
/// recorder.write(Record::from_scalar("reward", 1.0)); // No-op
/// recorder.flush(100); // No-op
/// ```
pub struct NullRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Phantom data to hold the type parameters
    phantom: PhantomData<(E, R)>,
}

impl<E, R> NullRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Creates a new null recorder.
    ///
    /// # Returns
    ///
    /// A new instance of `NullRecorder` that will discard all records.
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<E, R> Recorder<E, R> for NullRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Discards the given record without storing it.
    ///
    /// This method is a no-op and immediately discards the record.
    ///
    /// # Arguments
    ///
    /// * `_record` - The record to be discarded
    fn write(&mut self, _record: Record) {}

    /// Discards the given record without storing it.
    ///
    /// This method is a no-op and immediately discards the record.
    ///
    /// # Arguments
    ///
    /// * `_record` - The record to be discarded
    fn store(&mut self, _record: Record) {}

    /// No-op flush operation.
    ///
    /// This method does nothing as no records are stored.
    ///
    /// # Arguments
    ///
    /// * `_step` - The step at which to flush (ignored)
    fn flush(&mut self, _step: i64) {}
}
