use super::Record;

/// Writes a record to an output destination with [`Recorder::write`].
pub trait Recorder {
    /// Write a record to the [`Recorder`].
    fn write(&mut self, record: Record);

    /// Store the record.
    fn store(&mut self, record: Record);

    /// Writes values aggregated from the stored records.
    fn flush(&mut self, step: i64);
}
