#![allow(clippy::float_cmp)]
use std::cell::RefCell;
use log::info;
use crate::core::{
    Obs, Step, Env, Policy, record::{Record, Recorder, RecordValue}
};

/// Takes an action based on the policy and apply it to the environment.
///
/// It returns [crate::core::base::Step] object.
pub fn sample<E: Env, P: Policy<E>>(env: &mut E, policy: &mut P,
    obs_prev: &RefCell<Option<E::Obs>>) -> (Step<E>, Record) {
    let obs = obs_prev.replace(None)
        .expect("The buffer of the previous observations is not initialized.");
    let a = policy.sample(&obs);
    let (step, record) = env.step(&a);

    // Replace observations if the corresponding environments are resetted.
    let obs_reset = env.reset(Some(&step.is_done)).unwrap();
    let obs_reset = step.obs.clone().merge(obs_reset, &step.is_done);
    obs_prev.replace(Some(obs_reset));

    (step, record)
}

/// Run episodes with a policy and return cumlative rewards.
///
/// This method assumes that the environment is non-vectorized or `n_proc`=1.
pub fn eval<E: Env, P: Policy<E>>(env: &mut E, policy: &mut P, n_episodes: usize) -> Vec<f32> {
    // TODO: check the maximum number of steps of the environment for evaluation.
    // If it is infinite, the number of evaluation steps should be given in place of
    // n_episodes.
    let mut rs = Vec::new();
    let obs = env.reset(None).unwrap();
    let obs_prev = RefCell::new(Some(obs));

    for i in 0..n_episodes {
        let mut r_sum = 0.0;
        let mut steps = 0;
        loop {
            let (step, _) = sample(env, policy, &obs_prev);
            r_sum += &step.reward[0];
            if step.is_done[0] == 1.0 as f32 {
                break;
            }
            else {
                steps += 1;
            }
        }
        rs.push(r_sum);
        info!("Episode {:?}, {:?} steps", i, steps);
    }
    rs
}

/// Run episodes with a policy and recorder.
///
/// This method assumes that the environment is non-vectorized or `n_proc`=1.
pub fn eval_with_recorder<E, P, R>(env: &mut E, policy: &mut P, n_episodes: usize, recorder: &mut R) -> Vec<f32>
    where
    E: Env,
    P: Policy<E>,
    R: Recorder
{
    let mut rs = Vec::new();
    let obs = env.reset(None).unwrap();
    let obs_prev = RefCell::new(Some(obs));

    for episode in 0..n_episodes {
        let mut count_step = 0;
        let mut r_sum = 0.0;
        loop {
            let (step, mut record) = sample(env, policy, &obs_prev);
            if step.is_done[0] == 1.0 as f32 { break; }
            r_sum += &step.reward[0];

            record.insert("reward", RecordValue::Scalar(step.reward[0] as _));
            record.insert("episode", RecordValue::Scalar(episode as _));
            record.insert("step", RecordValue::Scalar(count_step as _));
            recorder.write(record);
            count_step += 1;
        }
        rs.push(r_sum);
    }
    rs
}