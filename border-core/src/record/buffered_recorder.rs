use super::{Record, Recorder};
use crate::{Env, ReplayBufferBase};
use std::marker::PhantomData;

/// Buffered recorder.
///
/// This is used for recording sequences of observation and action
/// during evaluation runs.
#[derive(Default)]
pub struct BufferedRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    buf: Vec<Record>,
    phantom: PhantomData<(E, R)>,
}

impl<E, R> BufferedRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Construct the recorder.
    pub fn new() -> Self {
        Self {
            buf: Vec::default(),
            phantom: PhantomData,
        }
    }

    /// Returns an iterator over the records.
    pub fn iter(&self) -> std::slice::Iter<Record> {
        self.buf.iter()
    }
}

impl<E, R> Recorder<E, R> for BufferedRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Write a [`Record`] to the buffer.
    ///
    /// TODO: Consider if it is worth making the method public.
    fn write(&mut self, record: Record) {
        self.buf.push(record);
    }

    fn flush(&mut self, _step: i64) {
        unimplemented!();
    }

    fn store(&mut self, _record: Record) {
        unimplemented!();
    }
}
