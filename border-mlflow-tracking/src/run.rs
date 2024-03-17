use serde::Deserialize;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct Run {
    pub info: RunInfo,
    data: Option<RunData>,
    inputs: Option<RunInputs>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub run_name: String,
    experiment_id: String,
    status: Option<String>,
    start_time: i64,
    end_time: Option<i64>,
    artifact_uri: Option<String>,
    lifecycle_stage: Option<String,>
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct RunData {
    metrics: Option<Vec<Metric>>,
    params: Option<Vec<Param>>,
    tags: Option<Vec<RunTag>>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
/// TODO: implement
struct RunInputs {}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct RunTag {
    key: String,
    value: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Param {
    key: String,
    value: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Metric {
    key: String,
    value: String,
    timestamp: i64,
    step: i64,
}
