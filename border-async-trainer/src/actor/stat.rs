use std::time::Duration;

/// Stats of sampling process in each [Actor](crate::Actor).
#[derive(Clone, Debug)]
pub struct ActorStat {
    /// The number of steps for interaction between agent and env.
    pub env_steps: usize,

    /// Duration of sampling loop in [Actor](crate::Actor).
    pub duration: Duration,
}

/// Returns a formatted string of the set of [ActorStat] for reporting.
pub fn actor_stats_fmt(stats: &Vec<ActorStat>) -> String {
    let mut s = "actor id, samples, duration [sec], samples per sec\n".to_string();
    for (i, stat) in stats.iter().enumerate() {
        let n = stat.env_steps;
        let d = stat.duration.as_secs_f32();
        let p = (n as f32) / d;
        s += format!("{}, {}, {}, {}\n", i, n, p, d).as_str();
    }
    s
}
