use super::{AggregateRecorder, Record, Recorder};

/// A recorder that ignores any record. This struct is used just for debugging.
pub struct NullRecorder {}

impl NullRecorder {}

impl Recorder for NullRecorder {
    /// Discard the given record.
    fn write(&mut self, _record: Record) {}
}

impl AggregateRecorder for NullRecorder {
    fn store(&mut self, _record: Record) {}

    fn flush(&mut self) {}
}
