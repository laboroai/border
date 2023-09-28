use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration of [AsyncTrainer](crate::AsyncTrainer)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AsyncTrainerConfig {
    /// Where to save the trained model.
    pub model_dir: Option<String>,

    /// Interval of recording in training steps.
    pub record_interval: usize,

    /// Interval of evaluation in training steps.
    pub eval_interval: usize,

    /// The maximal number of training steps.
    pub max_train_steps: usize,

    /// Interval of saving the model in optimization steps.
    pub save_interval: usize,

    /// Interval of synchronizing model parameters in training steps.
    pub sync_interval: usize,

    /// The number of episodes for evaluation
    pub eval_episodes: usize,

    /// Number of replay buffer divisions
    pub n_div_replaybuffer: usize,

    /// Number of pushed_item divisions
    pub n_div_pushed_item: usize,

    /// capacity of channel between each actor-manager and async-trainer
    pub channel_capacity: usize,
}

impl AsyncTrainerConfig {
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
