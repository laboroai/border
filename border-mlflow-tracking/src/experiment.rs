use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ExperimentTag {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Deserialize)]
/// all fields taken from <https://mlflow.org/docs/latest/rest-api.html#mlflowexperiment>.
pub struct Experiment {
    pub experiment_id: String,
    pub name: String,
    pub artifact_location: String,
    pub lifecycle_stage: String,
    pub last_update_time: i64,
    pub creation_time: i64,
    pub tags: Option<Vec<ExperimentTag>>,
}
