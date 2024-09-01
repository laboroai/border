use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration of [`AsyncTrainer`](crate::AsyncTrainer).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AsyncTrainerConfig {
    /// The maximum number of optimization steps.
    pub max_opts: usize,

    /// Where to save the trained model.
    pub model_dir: Option<String>,

    /// Interval of evaluation in training steps.
    pub eval_interval: usize,

    /// Interval of flushing records in optimization steps.
    pub flush_record_interval: usize,

    /// Interval of recording agent information in optimization steps.
    pub record_compute_cost_interval: usize,

    /// Interval of saving the model in optimization steps.
    pub save_interval: usize,

    /// Interval of synchronizing model parameters in training steps.
    pub sync_interval: usize,

    /// Warmup period, for filling replay buffer, in environment steps
    pub warmup_period: usize,
}

impl AsyncTrainerConfig {
    /// Sets the number of optimization steps.
    pub fn max_opts(mut self, v: usize) -> Result<Self> {
        self.max_opts = v;
        Ok(self)
    }

    /// Sets the interval of evaluation in optimization steps.
    pub fn eval_interval(mut self, v: usize) -> Result<Self> {
        self.eval_interval = v;
        Ok(self)
    }

    /// Sets the directory the trained model being saved.
    pub fn model_dir<T: Into<String>>(mut self, model_dir: T) -> Result<Self> {
        self.model_dir = Some(model_dir.into());
        Ok(self)
    }

    /// Sets the interval of computation cost in optimization steps.
    pub fn record_compute_cost_interval(
        mut self,
        record_compute_cost_interval: usize,
    ) -> Result<Self> {
        self.record_compute_cost_interval = record_compute_cost_interval;
        Ok(self)
    }

    /// Sets the interval of flushing recordd in optimization steps.
    pub fn flush_record_interval(mut self, flush_record_interval: usize) -> Result<Self> {
        self.flush_record_interval = flush_record_interval;
        Ok(self)
    }

    /// Sets warmup period in environment steps.
    pub fn warmup_period(mut self, warmup_period: usize) -> Result<Self> {
        self.warmup_period = warmup_period;
        Ok(self)
    }

    /// Sets the interval of saving in optimization steps.
    pub fn save_interval(mut self, save_interval: usize) -> Result<Self> {
        self.save_interval = save_interval;
        Ok(self)
    }

    /// Sets the interval of synchronizing model parameters in training steps.
    pub fn sync_interval(mut self, sync_interval: usize) -> Result<Self> {
        self.sync_interval = sync_interval;
        Ok(self)
    }

    /// Constructs [AsyncTrainerConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [AsyncTrainerConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

impl Default for AsyncTrainerConfig {
    /// There is no special intention behind these initial values.
    fn default() -> Self {
        Self {
            max_opts: 10, //000,
            model_dir: None,
            eval_interval: 5000,
            flush_record_interval: 5000,
            record_compute_cost_interval: 5000,
            save_interval: 50000,
            sync_interval: 100,
            warmup_period: 10000,
        }
    }
}
