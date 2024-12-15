use super::Record;
use anyhow::Result;
use std::path::Path;

/// Writes a record to an output destination with [`Recorder::write`].
pub trait Recorder {
    /// Write a record to the [`Recorder`].
    fn write(&mut self, record: Record);

    /// Store the record.
    fn store(&mut self, record: Record);

    /// Writes values aggregated from the stored records.
    fn flush(&mut self, step: i64);

    /// Saves the parameters of the given agent.
    #[allow(unused_variables)]
    fn save_params(&self, path: &Path) -> Result<()> {
        unimplemented!();
    }
}
