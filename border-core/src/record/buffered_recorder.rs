//! Buffered recorder implementation for sequential data recording.
//!
//! This module provides a buffered recorder that temporarily stores sequences of
//! observations and actions during evaluation runs. It is particularly useful
//! when you need to collect and analyze sequential data from reinforcement
//! learning environments.

use super::{Record, Recorder};
use crate::{Env, ReplayBufferBase};
use std::marker::PhantomData;

/// A recorder that buffers sequences of observations and actions in memory.
///
/// The `BufferedRecorder` is designed to temporarily store sequences of
/// environment interactions during evaluation runs. This is particularly useful
/// for analyzing agent behavior, debugging policies, or collecting demonstration
/// data.
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
///     record::{BufferedRecorder, Record},
///     Env, ReplayBufferBase,
/// };
///
/// // Create a new buffered recorder
/// let recorder = BufferedRecorder::<MyEnv, MyReplayBuffer>::new();
///
/// // Records can be added and later accessed for analysis
/// recorder.write(Record::from_scalar("reward", 1.0));
/// for record in recorder.iter() {
///     // Process each record
/// }
/// ```
#[derive(Default)]
pub struct BufferedRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// The internal buffer storing the sequence of records
    buf: Vec<Record>,
    /// Phantom data to hold the type parameters
    phantom: PhantomData<(E, R)>,
}

impl<E, R> BufferedRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Creates a new empty buffered recorder.
    ///
    /// # Returns
    ///
    /// A new instance of `BufferedRecorder` with an empty buffer.
    pub fn new() -> Self {
        Self {
            buf: Vec::default(),
            phantom: PhantomData,
        }
    }

    /// Returns an iterator over the recorded data.
    ///
    /// This method allows you to iterate over all records stored in the buffer
    /// without consuming them.
    ///
    /// # Returns
    ///
    /// An iterator over references to the [`Record`]s in the buffer.
    pub fn iter(&self) -> std::slice::Iter<Record> {
        self.buf.iter()
    }
}

impl<E, R> Recorder<E, R> for BufferedRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Writes a [`Record`] to the internal buffer.
    ///
    /// This method appends the given record to the end of the buffer. The records
    /// maintain their insertion order, which is important for sequential data analysis.
    ///
    /// # Arguments
    ///
    /// * `record` - The record to be stored in the buffer
    ///
    /// # Note
    ///
    /// This method is currently private as part of the [`Recorder`] trait implementation.
    /// Future versions may make it public if there's a need for direct record insertion.
    fn write(&mut self, record: Record) {
        self.buf.push(record);
    }

    /// Flushes the buffered data at the specified step.
    ///
    /// This method is currently unimplemented as the buffered recorder
    /// is designed to hold all data in memory until explicitly processed.
    ///
    /// # Arguments
    ///
    /// * `_step` - The step at which to flush the buffer
    ///
    /// # Panics
    ///
    /// This method will panic if called, as it is not implemented.
    fn flush(&mut self, _step: i64) {
        unimplemented!();
    }

    /// Stores a record in the underlying storage system.
    ///
    /// This method is currently unimplemented as the buffered recorder
    /// only supports in-memory storage.
    ///
    /// # Arguments
    ///
    /// * `_record` - The record to be stored
    ///
    /// # Panics
    ///
    /// This method will panic if called, as it is not implemented.
    fn store(&mut self, _record: Record) {
        unimplemented!();
    }
}
