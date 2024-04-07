use super::Record;

/// Process records provided with [`Recorder::write`]
pub trait Recorder {
    /// Write a record to the [`Recorder`].
    fn write(&mut self, record: Record);
}

/// Aggregates stored values, then writes.
pub trait AggregateRecorder {
    /// Store the record.
    fn store(&mut self, record: Record);

    /// Writes values aggregated from the stored records.
    fn flush(&mut self);
}
