use crate::{system_time_as_millis, Experiment, MlflowTrackingRecorder, Run};
use anyhow::Result;
use border_core::{Env, ReplayBufferBase};
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

#[derive(Clone, Debug, Deserialize)]
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
/// Request body of [Create Run](https://mlflow.org/docs/2.11.3/rest-api.html#id74).
struct CreateRunParams {
    experiment_id: String,
    start_time: i64,
    run_name: String,
}

#[derive(Debug, Serialize)]
/// Request body of [Create Experiment](https://mlflow.org/docs/2.11.3/rest-api.html#id67).
struct CreateExperimentParams {
    name: String,
}

#[derive(Debug, Serialize)]
/// Request body of [Search Runs](https://mlflow.org/docs/2.11.3/rest-api.html#id87).
struct SearchRunsParams {
    experiment_ids: Vec<String>,
    filter: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SearchRunsResponse {
    runs: Option<Vec<Run>>,
    #[allow(dead_code)]
    next_page_token: Option<String>,
}

#[derive(Debug)]
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

    /// Sets user name and password for basic authentication of the tracking server.
    pub fn basic_auth(self, user_name: impl AsRef<str>, password: impl AsRef<str>) -> Self {
        Self {
            client: self.client,
            base_url: self.base_url,
            experiment_id: self.experiment_id,
            user_name: user_name.as_ref().to_string(),
            password: password.as_ref().to_string(),
        }
    }

    /// Sets an experiment with the given name to this struct.
    pub fn set_experiment(self, name: impl AsRef<str>) -> Result<Self, GetExperimentIdError> {
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
    }

    /// Gets [`Experiment`] by name from the tracking server.
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
    /// If a Run with `name` exists in the tracking server, the run is used
    /// to create the recorder. If two or more runs with `name` exists,
    /// this method panics.
    ///
    /// You need to set an experiment using [`MlflowTrackingClient::set_experiment()`]
    /// before calling this method.
    ///
    /// This method uses `MLFLOW_DEFAULT_ARTIFACT_ROOT` environment variable as the directory
    /// where artifacts, like model parameters, will be saved. It is recommended to set this
    /// environment variable `mlruns` directory to which the tracking server persists experiment
    /// and run data.
    pub fn create_recorder<E, R>(
        &self,
        run_name: impl AsRef<str>,
    ) -> Result<MlflowTrackingRecorder<E, R>>
    where
        E: Env,
        R: ReplayBufferBase,
    {
        let run_name = run_name.as_ref();
        let run = {
            let runs = self.get_runs_by_name(run_name)?;
            if runs.len() >= 2 {
                panic!("There are 2 or more runs with name '{}'", run_name);
            } else if runs.len() == 1 {
                runs[0].clone()
            } else {
                self.get_run_info(run_name)?
            }
        };
        if run_name.len() == 0 {
            info!(
                "Run name '{}' has been automatically generated",
                run.info.run_name
            );
        }

        // Get the directory to which artifacts will be saved
        let artifact_base = crate::get_artifact_base(run.clone())?;

        // Return a recorder
        let experiment_id = self.experiment_id.as_ref().expect("Needs experiment_id");
        MlflowTrackingRecorder::new(&self.base_url, &experiment_id, run, artifact_base)
    }

    /// Get Run info.
    fn get_run_info(&self, run_name: impl AsRef<str>) -> Result<Run> {
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
        Ok(run)
    }

    /// Get runs by names.
    ///
    /// This method queries the tracking server and returns [`Run`]s.
    pub fn get_runs_by_name(&self, name: impl AsRef<str>) -> Result<Vec<Run>> {
        let experiment_id = self
            .experiment_id
            .clone()
            .expect("Experiment id must be set before search runs");
        let resp = self
            .post(
                self.url("runs/search"),
                &SearchRunsParams {
                    experiment_ids: vec![experiment_id],
                    filter: format!("tags.mlflow.runName = '{}'", name.as_ref()),
                },
            )
            .unwrap();

        let resp: SearchRunsResponse =
            serde_json::from_str(&resp.text().unwrap().as_str()).expect("Failed to deserialize");

        Ok(resp.runs.unwrap_or(vec![]))
    }
}

// // Used to test on vscode
// #[test]
// fn test() -> Result<()> {
//     let client = MlflowTrackingClient::new("http://localhost:8080").set_experiment("Gym")?;
//     // let runs = client.get_runs_by_name("dqn_cartpole_candle")?;
//     let runs = client.get_runs_by_name("")?;
//     println!("{:?}", runs.len());

//     Ok(())
// }
