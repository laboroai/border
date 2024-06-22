use super::{Record, Recorder};

/// Buffered recorder.
///
/// This is used for recording sequences of observation and action
/// during evaluation runs.
#[derive(Default)]
pub struct BufferedRecorder(Vec<Record>);

impl BufferedRecorder {
    /// Construct the recorder.
    pub fn new() -> Self {
        Self(Vec::default())
    }

    /// Returns an iterator over the records.
    pub fn iter(&self) -> std::slice::Iter<Record> {
        self.0.iter()
    }
}

impl Recorder for BufferedRecorder {
    /// Write a [`Record`] to the buffer.
    ///
    /// TODO: Consider if it is worth making the method public.
    fn write(&mut self, record: Record) {
        self.0.push(record);
    }
}
