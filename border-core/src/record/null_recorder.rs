use std::marker::PhantomData;

use super::{Record, Recorder};
use crate::{Env, ReplayBufferBase};

/// A recorder that ignores any record. This struct is used just for debugging.
pub struct NullRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    phantom: PhantomData<(E, R)>,
}

impl<E, R> NullRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    #[allow(missing_docs)]
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
    /// Discard the given record.
    fn write(&mut self, _record: Record) {}

    fn store(&mut self, _record: Record) {}

    fn flush(&mut self, _step: i64) {}
}
