use crate::{system_time_as_millis, Run};
use anyhow::Result;
use border_core::{
    record::{RecordStorage, RecordValue, Recorder},
    Env, ReplayBufferBase,
};
use chrono::{DateTime, Duration, Local, SecondsFormat};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::Value;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tempdir::TempDir;

#[derive(Debug, Serialize)]
struct LogParamParams<'a> {
    run_id: &'a String,
    key: &'a String,
    value: String,
}

#[derive(Debug, Serialize)]
struct LogMetricParams<'a> {
    run_id: &'a String,
    key: &'a String,
    value: f64,
    timestamp: i64,
    step: i64,
}

#[derive(Debug, Serialize)]
struct UpdateRunParams<'a> {
    run_id: &'a String,
    status: String,
    end_time: i64,
    run_name: &'a String,
}

#[derive(Debug, Serialize)]
struct SetTagParams<'a> {
    run_id: &'a String,
    key: &'a String,
    value: &'a String,
}

#[allow(dead_code)]
/// Record metrics to the MLflow tracking server during training.
///
/// Before training, you can use [`MlflowTrackingRecorder::log_params()`] to log parameters
/// of the run like hyperparameters of the algorithm, the name of environment on which the
/// agent is trained, etc.
///
/// [`MlflowTrackingRecorder::write()`] method logs [`RecordValue::Scalar`] values in the record
/// as metrics. As an exception, `opt_steps` is treated as the `step` field of Mlflow's metric data
/// (<https://mlflow.org/docs/latest/rest-api.html#metric>).
///
/// Other types of values like [`RecordValue::Array1`] will be ignored.
///
/// When dropped, this struct updates run's status to "FINISHED"
/// (<https://mlflow.org/docs/latest/rest-api.html#mlflowrunstatus>).
///
/// [`RecordValue::Scalar`]: border_core::record::RecordValue::Scalar
/// [`RecordValue::Array1`]: border_core::record::RecordValue::Array1
pub struct MlflowTrackingRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    client: Client,
    base_url: String,
    experiment_id: String,
    run_id: String,
    run_name: String,
    user_name: String,
    storage: RecordStorage,
    password: String,
    start_time: DateTime<Local>,
    artifact_base: PathBuf,
    phantom: PhantomData<(E, R)>,
}

impl<E, R> MlflowTrackingRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Create a new instance of `MlflowTrackingRecorder`.
    ///
    /// This method is used in [`MlflowTrackingClient::create_recorder()`].
    ///
    /// This method adds a tag "host_start_time" with the current time.
    /// This tag is useful when using mlflow-export-import: it losts the original time.
    /// See https://github.com/mlflow/mlflow-export-import/issues/72
    ///
    /// [`MlflowTrackingClient::create_recorder()`]: crate::MlflowTrackingClient::create_recorder
    pub fn new(
        base_url: &String,
        experiment_id: &String,
        run: &Run,
        artifact_base: PathBuf,
    ) -> Result<Self> {
        let client = Client::new();
        let start_time = Local::now();
        let recorder = Self {
            client,
            base_url: base_url.clone(),
            experiment_id: experiment_id.to_string(),
            run_id: run.info.run_id.clone(),
            run_name: run.info.run_name.clone(),
            user_name: "".to_string(),
            password: "".to_string(),
            storage: RecordStorage::new(),
            start_time: start_time.clone(),
            artifact_base,
            phantom: PhantomData,
        };

        // Record current time as tag "host_start_time"
        recorder.set_tag(
            "host_start_time",
            start_time.to_rfc3339_opts(SecondsFormat::Secs, true),
        )?;

        Ok(recorder)
    }

    pub fn log_params(&self, params: impl Serialize) -> Result<()> {
        let url = format!("{}/api/2.0/mlflow/runs/log-parameter", self.base_url);
        let flatten_map = {
            let map = match serde_json::to_value(params).unwrap() {
                Value::Object(map) => map,
                _ => panic!("Failed to parse object"),
            };
            flatten_serde_json::flatten(&map)
        };
        for (key, value) in flatten_map.iter() {
            let params = LogParamParams {
                run_id: &self.run_id,
                key,
                value: value.to_string(),
            };
            let _resp = self
                .client
                .post(&url)
                .basic_auth(&self.user_name, Some(&self.password))
                .json(&params) // auto serialize
                .send()
                .unwrap();
            // TODO: error handling caused by API call
        }

        Ok(())
    }

    pub fn set_tag(&self, key: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
        let url = format!("{}/api/2.0/mlflow/runs/set-tag", self.base_url);
        let params = SetTagParams {
            run_id: &self.run_id,
            key: &key.as_ref().to_string(),
            value: &value.as_ref().to_string(),
        };
        let _resp = self
            .client
            .post(&url)
            .basic_auth(&self.user_name, Some(&self.password))
            .json(&params)
            .send()
            .unwrap();

        Ok(())
    }
}

impl<E, R> Recorder<E, R> for MlflowTrackingRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    fn write(&mut self, record: border_core::record::Record) {
        let url = format!("{}/api/2.0/mlflow/runs/log-metric", self.base_url);
        let timestamp = system_time_as_millis() as i64;
        let step = record.get_scalar("opt_steps").unwrap() as i64;

        for (key, value) in record.iter() {
            if *key != "opt_steps" {
                match value {
                    RecordValue::Scalar(v) => {
                        let value = *v as f64;
                        let params = LogMetricParams {
                            run_id: &self.run_id,
                            key,
                            value,
                            timestamp,
                            step,
                        };
                        let _resp = self
                            .client
                            .post(&url)
                            .basic_auth(&self.user_name, Some(&self.password))
                            .json(&params) // auto serialize
                            .send()
                            .unwrap();
                        // TODO: error handling caused by API call
                    }
                    _ => {} // ignore record value
                }
            }
        }
    }

    fn flush(&mut self, step: i64) {
        let mut record = self.storage.aggregate();
        record.insert("opt_steps", RecordValue::Scalar(step as _));
        self.write(record);
    }

    fn store(&mut self, record: border_core::record::Record) {
        self.storage.store(record);
    }

    /// Save the model parameters as artifacts of MLflow.
    ///
    /// MLflow server is assumed to be running on the same host as the program using this struct.
    /// Under this condition, this method saves model parameters under the `mlruns` directory managed by
    /// the MLflow server. This method recognizes the environment variable `MLFLOW_DEFAULT_ARTIFACT_ROOT`
    /// as the location of the `mlruns` directory.
    fn save_model(&self, base: &Path, agent: &Box<dyn border_core::Agent<E, R>>) -> Result<()> {
        // Saves the artifacts in the temporary directory
        let tmp = TempDir::new("mlflow")?;
        let srcs = agent.save_params(&tmp.path())?;

        // Copies the artifacts
        for src in srcs.iter() {
            let dest = {
                // Create subdirectory
                let path = self.artifact_base.join(base);
                if !path.exists() {
                    let _ = std::fs::create_dir(&path)?;
                }

                // Create destination file path
                let file = src.strip_prefix(tmp.path())?;
                path.join(file)
            };
            let bytes = std::fs::copy(src, &dest)?;
            log::info!("Save {:?}", &src);
            log::info!("Copy {:?}, {:.2}MB", &dest, bytes as f32 / (1024. * 1024.));
        }
        Ok(())
    }
}

impl<E, R> Drop for MlflowTrackingRecorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Update run's status to "FINISHED" when dropped.
    ///
    /// It also adds tags "host_end_time" and "host_duration" with the current time and duration.
    fn drop(&mut self) {
        let end_time = Local::now();
        let duration = end_time.signed_duration_since(self.start_time);
        self.set_tag(
            "host_end_time",
            end_time.to_rfc3339_opts(SecondsFormat::Secs, true),
        )
        .unwrap();
        self.set_tag("host_duration", format_duration(&duration))
            .unwrap();

        let url = format!("{}/api/2.0/mlflow/runs/update", self.base_url);
        let params = UpdateRunParams {
            run_id: &self.run_id,
            status: "FINISHED".to_string(),
            end_time: end_time.timestamp_millis(),
            run_name: &self.run_name,
        };
        let _resp = self
            .client
            .post(&url)
            .basic_auth(&self.user_name, Some(&self.password))
            .json(&params) // auto serialize
            .send()
            .unwrap();
        // TODO: error handling caused by API call
    }
}

fn format_duration(dt: &Duration) -> String {
    let mut seconds = dt.num_seconds();
    let mut minutes = seconds / 60;
    seconds %= 60;
    let hours = minutes / 60;
    minutes %= 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}
