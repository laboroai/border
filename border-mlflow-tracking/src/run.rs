use serde::Deserialize;

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
pub struct Run {
    pub info: RunInfo,
    data: Option<RunData>,
    inputs: Option<RunInputs>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub run_name: String,
    pub experiment_id: String,
    pub status: Option<String>,
    pub start_time: i64,
    pub end_time: Option<i64>,
    pub artifact_uri: Option<String>,
    pub lifecycle_stage: Option<String>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct RunData {
    metrics: Option<Vec<Metric>>,
    params: Option<Vec<Param>>,
    tags: Option<Vec<RunTag>>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
/// TODO: implement
struct RunInputs {}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct RunTag {
    key: String,
    value: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct Param {
    key: String,
    value: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct Metric {
    key: String,
    value: f64,
    timestamp: i64,
    step: i64,
}
