// use anyhow::Result;
use crate::{system_time_as_millis, Experiment, MlflowTrackingRecorder, Run};
use anyhow::Result;
use log::info;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::Display;

#[derive(Debug, Deserialize)]
/// Internally used.
struct Experiment_ {
    pub(crate) experiment: Experiment,
}

#[derive(Debug, Deserialize)]
/// Internally used.
struct Run_ {
    run: Run,
}

#[derive(Debug, Clone)]
pub struct GetExperimentIdError;

impl Display for GetExperimentIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to get experiment ID")
    }
}

impl Error for GetExperimentIdError {}

#[derive(Debug, Serialize)]
/// Parameters adapted from https://mlflow.org/docs/latest/rest-api.html#id74
///
/// TODO: Support parameters in API, if required.
struct CreateRunParams {
    experiment_id: String,
    start_time: i64,
    run_name: String,
}

/// Provides access to a MLflow tracking server via REST API.
///
/// Support Mlflow API version 2.0.
pub struct MlflowTrackingClient {
    client: Client,

    /// Base URL.
    base_url: String,

    /// Current experiment ID.
    experiment_id: Option<String>,
}

impl MlflowTrackingClient {
    pub fn new(base_url: impl AsRef<str>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.as_ref().to_string(),
            experiment_id: None,
        }
    }

    /// Set the experiment ID to this struct.
    ///
    /// Get the ID from the tracking server by `name`.
    pub fn set_experiment_id(self, name: impl AsRef<str>) -> Result<Self, GetExperimentIdError> {
        let experiment_id = {
            self.get_experiment(name.as_ref())
                .expect(format!("Failed to get experiment: {:?}", name.as_ref()).as_str())
                .experiment_id
        };

        info!(
            "For experiment '{}', id={} is set in MlflowTrackingClient",
            name.as_ref(),
            experiment_id
        );

        Ok(Self {
            client: self.client,
            base_url: self.base_url,
            experiment_id: Some(experiment_id),
        })

        // let experiment_id = self.get_experiment_id(&name);
        // match experiment_id {
        //     None => Err(GetExperimentIdError),
        //     Some(experiment_id) => Ok(Self {
        //         client: self.client,
        //         base_url: self.base_url,
        //         experiment_id: Some(experiment_id),
        //     }),
        // }
    }

    /// Get [`Experiment`] by name from the tracking server.
    ///
    /// TODO: Better error handling
    pub fn get_experiment(&self, name: impl AsRef<str>) -> Option<Experiment> {
        let url = format!("{}/api/2.0/mlflow/experiments/get-by-name", self.base_url);
        let resp = self
            .client
            .get(url)
            .query(&[("experiment_name", name.as_ref())])
            .send()
            .unwrap();
        let experiment: Experiment_ = serde_json::from_str(resp.text().unwrap().as_str()).unwrap();

        Some(experiment.experiment)
    }

    /// Create [`MlflowTrackingRecorder`] corresponding to a run.
    ///
    /// If `name` is empty (`""`), a run name is given by the tracking server.
    ///
    /// Need to call [`MlflowTrackingClient::set_experiment_id()`] before calling this method.
    pub fn create_recorder(&self, run_name: impl AsRef<str>) -> Result<MlflowTrackingRecorder> {
        let not_given_name = run_name.as_ref().len() == 0;
        let experiment_id = self.experiment_id.as_ref().expect("Needs experiment_id");
        let url = format!("{}/api/2.0/mlflow/runs/create", self.base_url);
        let params = CreateRunParams {
            experiment_id: experiment_id.to_string(),
            start_time: system_time_as_millis() as i64,
            run_name: run_name.as_ref().to_string(),
        };
        let resp = self
            .client
            .post(url)
            .json(&params) // auto serialize
            .send()
            .unwrap();
        // println!("{:?}", resp);
        // println!("{:?}", resp.text());
        // TODO: Check the response from the tracking server here
        let run = {
            let run: Run_ =
                serde_json::from_str(&resp.text().unwrap()).expect("Failed to deserialize Run");
            run.run
        };
        if not_given_name {
            info!(
                "Run name '{}' has been automatically generated",
                run.info.run_name
            );
        }
        MlflowTrackingRecorder::new(&self.base_url, &experiment_id, &run)
    }
}
