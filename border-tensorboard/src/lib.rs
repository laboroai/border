use anyhow::Result;
use border_core::{
    record::{Record, RecordValue, Recorder},
    Env, ReplayBufferBase,
};
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
};
use tensorboard_rs::summary_writer::SummaryWriter;

/// Write records to TFRecord.
pub struct TensorboardRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    model_dir: PathBuf,
    writer: SummaryWriter,
    step_key: String,
    latest_record: Option<Record>,
    ignore_unsupported_value: bool,
    phantom: PhantomData<(E, R)>,
}

impl<E, R> TensorboardRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Construct a [`TensorboardRecorder`].
    ///
    /// * `log_dir` - Directory in which TFRecords will be stored.
    /// * `model_dir` - Directory in which the trained model will be saved.
    /// * `check_unsupported_value` - If true, check unsupported record value in the write() method.
    pub fn new(
        log_dir: impl AsRef<Path>,
        model_dir: impl AsRef<Path>,
        check_unsupported_value: bool,
    ) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            writer: SummaryWriter::new(log_dir),
            step_key: "opt_steps".to_string(),
            ignore_unsupported_value: !check_unsupported_value,
            latest_record: None,
            phantom: PhantomData,
        }
    }
}

impl<E, R> Recorder<E, R> for TensorboardRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Write a given [`Record`] into a TFRecord.
    ///
    /// This method handles [RecordValue::Scalar] and [RecordValue::DateTime] in the [`Record`].
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

    fn store(&mut self, record: Record) {
        self.latest_record = Some(record);
    }

    fn flush(&mut self, step: i64) {
        if self.latest_record.is_some() {
            let mut record = self.latest_record.take().unwrap();
            record.insert("opt_steps", RecordValue::Scalar(step as _));
            self.write(record);
        }
    }

    /// Save the model parameters in the local file system.
    fn save_model(&self, base: &Path, agent: &Box<dyn border_core::Agent<E, R>>) -> Result<()> {
        let path = self.model_dir.join(base);
        let _ = agent.save_params(&path)?;
        Ok(())
    }
}
