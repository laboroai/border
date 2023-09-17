#![allow(clippy::float_cmp)]
//! Utilities for interaction of agents and environments.
use crate::{
    record::{RecordValue, Recorder},
    Env, Policy
};
use anyhow::Result;

/// Runs episodes with a policy and recorder.
///
/// This function does not support vectorized environments.
#[deprecated]
pub fn eval_with_recorder<E, P, R>(
    env: &mut E,
    policy: &mut P,
    n_episodes: usize,
    recorder: &mut R,
) -> Result<Vec<f32>>
where
    E: Env,
    P: Policy<E>,
    R: Recorder,
{
    let mut rs = Vec::new();

    for episode in 0..n_episodes {
        // let mut prev_obs = env.reset(None)?;
        let mut prev_obs = env.reset_with_index(episode)?;
        let mut count_step = 0;
        let mut r_total = 0.0;

        loop {
            let act = policy.sample(&prev_obs);
            let (step, mut record) = env.step(&act);
            r_total += step.reward[0];
            prev_obs = step.obs;

            record.insert("reward", RecordValue::Scalar(step.reward[0] as _));
            record.insert("episode", RecordValue::Scalar(episode as _));
            record.insert("step", RecordValue::Scalar(count_step as _));
            recorder.write(record);

            if step.is_done[0] == 1 {
                break;
            }

            count_step += 1;
        }
        rs.push(r_total);
    }

    Ok(rs)
}

/// Runs environment steps with a given policy and recorder.
///
/// This function does not support vectorized environments.
/// 
/// * `n_steps` - The maximum number of environment steps.
///   The interaction loop is terminated when is_done is true before reaching `n_steps` environment steps.
/// * `prev_obs` - The observation, applied to the policy at the first step of interaction.
///   If `None`, `env.reset_with_index(0)` is invoked.
/// 
/// This function returns the sum of rewards during interaction.
#[deprecated]
pub fn eval_with_recorder2<E, P, R>(
    env: &mut E,
    policy: &mut P,
    n_steps: usize,
    recorder: &mut R,
    prev_obs: Option<<E as Env>::Obs>,
) -> Result<(f32, <E as Env>::Obs)>
where
    E: Env,
    P: Policy<E>,
    R: Recorder,
{
    let mut r_total = 0.0;
    let mut prev_obs = if prev_obs.is_some() {
        prev_obs.unwrap()
    } else {
        env.reset_with_index(0)?
    };

    for t in 0..n_steps {
        let act = policy.sample(&prev_obs);
        let (step, mut record) = env.step(&act);
        r_total += step.reward[0];
        prev_obs = step.obs;

        record.insert("reward", RecordValue::Scalar(step.reward[0] as _));
        record.insert("step", RecordValue::Scalar(t as _));
        recorder.write(record);

        if step.is_done[0] == 1 {
            break;
        }
    }

    Ok((r_total, prev_obs))
}
