use std::time::Duration;
/// Stats of [`AsyncTrainer`](crate::AsyncTrainer)`::train()`.
pub struct AsyncTrainStat {
    /// The number of samples pushed to the replay buffer per second.
    pub samples_per_sec: f32,

    /// Duration of training.
    pub duration: Duration,

    /// The number of optimization steps per second.
    pub opt_per_sec: f32,
}

impl AsyncTrainStat {
    /// Returns a formatted string.
    pub fn fmt(&self) -> String {
        let mut s = "samples/sec, opt_steps/sec, duration\n".to_string();
        s += format!(
            "{}, {}, {}\n",
            self.samples_per_sec,
            self.opt_per_sec,
            self.duration.as_secs_f32()
        )
        .as_str();
        s
    }
}
