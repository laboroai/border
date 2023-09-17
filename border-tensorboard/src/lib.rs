use border_core::record::{Record, RecordValue, Recorder};
use std::path::Path;
use tensorboard_rs::summary_writer::SummaryWriter;

/// Write records to TFRecord.
pub struct TensorboardRecorder {
    writer: SummaryWriter,
    step_key: String,
    ignore_unsupported_value: bool,
}

impl TensorboardRecorder {
    /// Construct a [`TensorboardRecorder`].
    ///
    /// TFRecord will be stored in `logdir`.
    pub fn new<P: AsRef<Path>>(logdir: P) -> Self {
        Self {
            writer: SummaryWriter::new(logdir),
            step_key: "opt_steps".to_string(),
            ignore_unsupported_value: true,
        }
    }

    /// Construct a [`TensorboardRecorder`] with checking unsupported record value.
    ///
    /// TFRecord will be stored in `logdir`.
    pub fn new_with_check_unsupported_value<P: AsRef<Path>>(logdir: P) -> Self {
        Self {
            writer: SummaryWriter::new(logdir),
            step_key: "opt_steps".to_string(),
            ignore_unsupported_value: false,
        }
    }
}

impl Recorder for TensorboardRecorder {
    /// Write a given [Record] into a TFRecord.
    ///
    /// This method handles [RecordValue::Scalar] and [RecordValue::DateTime] in the [Record].
    /// Other variants will be ignored.
    fn write(&mut self, record: Record) {
        // TODO: handle error
        let step = match record.get(&self.step_key).unwrap() {
            RecordValue::Scalar(v) => *v as usize,
            _ => {
                panic!()
            }
        };

        for (k, v) in record.iter() {
            if *k != self.step_key {
                match v {
                    RecordValue::Scalar(v) => self.writer.add_scalar(k, *v as f32, step),
                    RecordValue::DateTime(_) => {} // discard value
                    RecordValue::Array2(data, shape) => {
                        let shape = [3, shape[0], shape[1]];
                        let min = data.iter().fold(f32::MAX, |m, v| v.min(m));
                        let scale = data.iter().fold(-f32::MAX, |m, v| v.max(m)) - min;
                        let mut data = data
                            .iter()
                            .map(|&e| ((e - min) / scale * 255f32) as u8)
                            .collect::<Vec<_>>();
                        let data_ = data.clone();
                        data.extend(data_.iter());
                        data.extend(data_.iter());
                        self.writer.add_image(k, data.as_slice(), &shape, step)
                    }
                    _ => {
                        if !self.ignore_unsupported_value {
                            panic!("Unsupported value: {:?}", (k, v));
                        }
                    }
                };
            }
        }
    }
}
