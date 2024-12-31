//! A logger for border-core crate.
//!
//! This crate is based on [MLflow](https://mlflow.org) tracking.
//!
//! Before running the program using this crate, run a tracking server with the following command:
//!
//! ```bash
//! mlflow server --host 127.0.0.1 --port 8080
//! ```
//!
//! Then, training configurations and metrices can be logged to the tracking server.
//! The following code is an example. Nested configuration parameters will be flattened,
//! logged like `hyper_params.param1`, `hyper_params.param2`.
//!
//! ```no_run
//! use anyhow::Result;
//! use border_core::record::{Record, RecordValue, Recorder};
//! use border_mlflow_tracking::MlflowTrackingClient;
//! use serde::Serialize;
//!
//! // Nested Configuration struct
//! #[derive(Debug, Serialize)]
//! struct Config {
//!     env_params: String,
//!     hyper_params: HyperParameters,
//! }
//!
//! #[derive(Debug, Serialize)]
//! struct HyperParameters {
//!     param1: i64,
//!     param2: Param2,
//!     param3: Param3,
//! }
//!
//! #[derive(Debug, Serialize)]
//! enum Param2 {
//!     Variant1,
//!     Variant2(f32),
//! }
//!
//! #[derive(Debug, Serialize)]
//! struct Param3 {
//!     dataset_name: String,
//! }
//!
//! fn main() -> Result<()> {
//!     env_logger::init();
//!
//!     let config1 = Config {
//!         env_params: "env1".to_string(),
//!         hyper_params: HyperParameters {
//!             param1: 0,
//!             param2: Param2::Variant1,
//!             param3: Param3 {
//!                 dataset_name: "a".to_string(),
//!             },
//!         },
//!     };
//!     let config2 = Config {
//!         env_params: "env2".to_string(),
//!         hyper_params: HyperParameters {
//!             param1: 0,
//!             param2: Param2::Variant2(3.0),
//!             param3: Param3 {
//!                 dataset_name: "a".to_string(),
//!             },
//!         },
//!     };
//!
//!     // Set experiment for runs
//!     let client = MlflowTrackingClient::new("http://localhost:8080").set_experiment("Default")?;
//!
//!     // Create recorders for logging
//!     let mut recorder_run1 = client.create_recorder("")?;
//!     let mut recorder_run2 = client.create_recorder("")?;
//!     recorder_run1.log_params(&config1)?;
//!     recorder_run2.log_params(&config2)?;
//!
//!     // Logging while training
//!     for opt_steps in 0..100 {
//!         let opt_steps = opt_steps as f32;
//!
//!         // Create a record
//!         let mut record = Record::empty();
//!         record.insert("opt_steps", RecordValue::Scalar(opt_steps));
//!         record.insert("Loss", RecordValue::Scalar((-1f32 * opt_steps).exp()));
//!
//!         // Log metrices in the record
//!         recorder_run1.write(record);
//!     }
//!
//!     // Logging while training
//!     for opt_steps in 0..100 {
//!         let opt_steps = opt_steps as f32;
//!
//!         // Create a record
//!         let mut record = Record::empty();
//!         record.insert("opt_steps", RecordValue::Scalar(opt_steps));
//!         record.insert("Loss", RecordValue::Scalar((-0.5f32 * opt_steps).exp()));
//!
//!         // Log metrices in the record
//!         recorder_run2.write(record);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Save model parameters during training
//!
//! [`MlflowTrackingClient`] relies on the `MLFLOW_DEFAULT_ARTIFACT_ROOT` environment variable
//! to locate where model parameters are saved during training. Note that this environment variable
//! should be set for the program using this crate, not for the tracking server program.
//! Currently, only saving to the local file system is supported.
//!
mod client;
mod experiment;
mod recorder;
mod run;
use anyhow::Result;
pub use client::{GetExperimentIdError, MlflowTrackingClient};
use experiment::Experiment;
pub use recorder::MlflowTrackingRecorder;
pub use run::Run;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Code adapted from <https://stackoverflow.com/questions/26593387>.
fn system_time_as_millis() -> u128 {
    let time = SystemTime::now();
    time.duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis()
}

/// Get the directory to which artifacts will be saved.
pub(crate) fn get_artifact_base(run: Run) -> Result<PathBuf> {
    let artifact_uri: PathBuf = run
        .clone()
        .info
        .artifact_uri
        .expect("Failed to get artifact_uri")
        .into();
    let artifact_uri = artifact_uri.strip_prefix("mlflow-artifacts:/")?;
    let path: PathBuf = std::env::var("MLFLOW_DEFAULT_ARTIFACT_ROOT")
        .expect("MLFLOW_DEFAULT_ARTIFACT_ROOT must be set")
        .into();
    Ok(path.join(artifact_uri))
}

// /// https://stackoverflow.com/questions/26958489/how-to-copy-a-folder-recursively-in-rust
// fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
//     fs::create_dir_all(&dst)?;
//     for entry in fs::read_dir(src)? {
//         let entry = entry?;
//         let ty = entry.file_type()?;
//         if ty.is_dir() {
//             copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
//         } else {
//             fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
//         }
//     }
//     Ok(())
// }
