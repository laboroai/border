use clap::Parser;
use serde::{Deserialize, Serialize};

/// Train/eval DQN agent in atari environment
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
#[command(version, about)]
pub struct Args {
    /// "train", "eval" or "show_config".
    /// In evaluation mode, the trained model is loaded.
    #[arg(long)]
    pub mode: String,

    /// Device name.
    /// If set to `"Cpu"`, the CPU will be used.
    /// Otherwise, the device will be determined by the `cuda_if_available()` method.
    #[arg(long)]
    pub device: Option<String>,

    /// Run name of MLflow.
    /// When using this option, an MLflow server must be running.
    /// If no name is provided, the log will be recorded in TensorBoard.
    #[arg(long)]
    pub mlflow_run_name: Option<String>,

    /// Waiting time in milliseconds between frames when evaluation
    #[arg(long, default_value_t = 25)]
    pub wait: u64,

    /// Name of the game
    pub name: String,
} 