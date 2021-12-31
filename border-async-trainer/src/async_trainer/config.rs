use serde::{Deserialize, Serialize};

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
}
