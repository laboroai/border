//! Configuration of [`Trainer`](super::Trainer).
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration of [`Trainer`](super::Trainer).
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct TrainerConfig {
    /// The maximum number of optimization steps.
    pub max_opts: usize,

    /// Interval of optimization steps in environment steps.
    pub opt_interval: usize,

    /// Interval of evaluation in optimization steps.
    pub eval_interval: usize,

    /// Interval of flushing records in optimization steps.
    pub flush_record_interval: usize,

    /// Interval of recording agent information in optimization steps.
    pub record_compute_cost_interval: usize,

    /// Interval of recording agent information in optimization steps.
    pub record_agent_info_interval: usize,

    /// Warmup period, for filling replay buffer, in environment steps
    pub warmup_period: usize,

    /// Intercal of saving model parameters in optimization steps.
    pub save_interval: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_opts: 0,
            eval_interval: 0,
            // eval_threshold: None,
            opt_interval: 1,
            flush_record_interval: usize::MAX,
            record_compute_cost_interval: usize::MAX,
            record_agent_info_interval: usize::MAX,
            warmup_period: 0,
            save_interval: usize::MAX,
        }
    }
}

impl TrainerConfig {
    /// Sets the number of optimization steps.
    pub fn max_opts(mut self, v: usize) -> Self {
        self.max_opts = v;
        self
    }

    /// Sets the interval of evaluation in optimization steps.
    pub fn eval_interval(mut self, v: usize) -> Self {
        self.eval_interval = v;
        self
    }

    /// (Deprecated) Sets the evaluation threshold.
    pub fn eval_threshold(/*mut */ self, _v: f32) -> Self {
        unimplemented!();
        // self.eval_threshold = Some(v);
        // self
    }

    /// Sets the interval of optimization in environment steps.
    pub fn opt_interval(mut self, opt_interval: usize) -> Self {
        self.opt_interval = opt_interval;
        self
    }

    /// Sets the interval of flushing recordd in optimization steps.
    pub fn flush_record_interval(mut self, flush_record_interval: usize) -> Self {
        self.flush_record_interval = flush_record_interval;
        self
    }

    /// Sets the interval of computation cost in optimization steps.
    pub fn record_compute_cost_interval(mut self, record_compute_cost_interval: usize) -> Self {
        self.record_compute_cost_interval = record_compute_cost_interval;
        self
    }

    /// Sets the interval of recording agent information in optimization steps.
    pub fn record_agent_info_interval(mut self, record_agent_info_interval: usize) -> Self {
        self.record_agent_info_interval = record_agent_info_interval;
        self
    }

    /// Sets warmup period in environment steps.
    pub fn warmup_period(mut self, warmup_period: usize) -> Self {
        self.warmup_period = warmup_period;
        self
    }

    /// Sets the interval of saving in optimization steps.
    pub fn save_interval(mut self, save_interval: usize) -> Self {
        self.save_interval = save_interval;
        self
    }

    /// Constructs [`TrainerConfig`] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [`TrainerConfig`].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tempdir::TempDir;

//     #[test]
//     fn test_serde_trainer_builder() -> Result<()> {
//         let builder = TrainerBuilder::default()
//             .max_opts(100)
//             .eval_interval(10000)
//             .n_episodes_per_eval(5)
//             .model_dir("some/directory");

//         let dir = TempDir::new("trainer_builder")?;
//         let path = dir.path().join("trainer_builder.yaml");
//         println!("{:?}", path);

//         builder.save(&path)?;
//         let builder_ = TrainerBuilder::load(&path)?;
//         assert_eq!(builder, builder_);
//         // let yaml = serde_yaml::to_string(&trainer)?;
//         // println!("{}", yaml);
//         // assert_eq!(
//         //     yaml,
//         //     "---\n\
//         //      max_opts: 100\n\
//         //      eval_interval: 10000\n\
//         //      n_episodes_per_eval: 5\n\
//         //      eval_threshold: ~\n\
//         //      model_dir: some/directory\n\
//         // "
//         // );
//         Ok(())
//     }
// }
