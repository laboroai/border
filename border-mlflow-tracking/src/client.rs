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

#[derive(Debug, Serialize)]
struct CreateExperimentParams {
    name: String,
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

    /// User name of the tracking server.
    user_name: String,

    /// Password.
    password: String,
}

impl MlflowTrackingClient {
    pub fn new(base_url: impl AsRef<str>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.as_ref().to_string(),
            experiment_id: None,
            user_name: "".to_string(),
            password: "".to_string(),
        }
    }

    /// Set user name and password for basic authentication of the tracking server.
    pub fn basic_auth(self, user_name: impl AsRef<str>, password: impl AsRef<str>) -> Self {
        Self {
            client: self.client,
            base_url: self.base_url,
            experiment_id: self.experiment_id,
            user_name: user_name.as_ref().to_string(),
            password: password.as_ref().to_string(),
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
            user_name: self.user_name,
            password: self.password,
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
    /// If the experiment with given name does not exist in the trackingserver,
    /// it will be created.
    ///
    /// TODO: Better error handling
    pub fn get_experiment(&self, name: impl AsRef<str>) -> Option<Experiment> {
        let resp = match self.get(
            self.url("experiments/get-by-name"),
            &[("experiment_name", name.as_ref())],
        ) {
            Ok(resp) => {
                if resp.status().is_success() {
                    resp
                } else {
                    // if the experiment does not exist, create it
                    self.post(
                        self.url("experiments/create"),
                        &CreateExperimentParams {
                            name: name.as_ref().into(),
                        },
                    )
                    .unwrap();
                    self.get(
                        self.url("experiments/get-by-name"),
                        &[("experiment_name", name.as_ref())],
                    )
                    .unwrap()
                }
            }
            Err(_) => {
                panic!();
            }
        };
        let experiment: Experiment_ = serde_json::from_str(resp.text().unwrap().as_str()).unwrap();

        Some(experiment.experiment)
    }

    fn url(&self, api: impl AsRef<str>) -> String {
        format!("{}/api/2.0/mlflow/{}", self.base_url, api.as_ref())
    }

    fn get(
        &self,
        url: String,
        query: &impl Serialize,
    ) -> reqwest::Result<reqwest::blocking::Response> {
        self.client
            .get(url)
            .basic_auth(&self.user_name, Some(&self.password))
            .query(query)
            .send()
    }

    fn post(
        &self,
        url: String,
        params: &impl Serialize,
    ) -> reqwest::Result<reqwest::blocking::Response> {
        self.client
            .post(url)
            .basic_auth(&self.user_name, Some(&self.password))
            .json(&params) // auto serialize
            .send()
    }

    /// Create [`MlflowTrackingRecorder`] corresponding to a run.
    ///
    /// If `name` is empty (`""`), a run name is generated by the tracking server.
    ///
    /// Need to call [`MlflowTrackingClient::set_experiment_id()`] before calling this method.
    pub fn create_recorder(&self, run_name: impl AsRef<str>) -> Result<MlflowTrackingRecorder> {
        let not_given_name = run_name.as_ref().len() == 0;
        let experiment_id = self.experiment_id.as_ref().expect("Needs experiment_id");
        let resp = self
            .post(
                self.url("runs/create"),
                &CreateRunParams {
                    experiment_id: experiment_id.to_string(),
                    start_time: system_time_as_millis() as i64,
                    run_name: run_name.as_ref().to_string(),
                },
            )
            .unwrap();
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
