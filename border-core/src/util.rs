#![allow(clippy::float_cmp)]
//! Utilities for interaction of agents and environments.
use crate::{
    record::{RecordValue, Recorder},
    Env, Policy
};
use anyhow::Result;

// /// Takes an action based on the policy and apply it to the environment.
// ///
// /// It returns [crate::core::base::Step] object.
// pub fn sample<E: Env, P: Policy<E>>(
//     env: &mut E,
//     policy: &mut P,
//     obs_prev: &RefCell<Option<E::Obs>>,
// ) -> (Step<E>, Record) {
//     let obs = obs_prev
//         .replace(None)
//         .expect("The buffer of the previous observations is not initialized.");
//     let a = policy.sample(&obs);
//     let (step, record) = env.step(&a);

//     // Replace observations if the corresponding environments are resetted.
//     let obs_reset = env.reset(Some(&step.is_done)).unwrap();
//     let obs_reset = step.obs.clone().merge(obs_reset, &step.is_done);
//     obs_prev.replace(Some(obs_reset));

//     (step, record)
// }

// #[cfg_attr(doc, aquamarine::aquamarine)]
// /// Run episodes with a policy and return cumlative rewards.
// ///
// /// This function assumes that the environment is non-vectorized or `n_proc`=1.
// ///
// /// In this function, the main entities of the library, i.e., an environment ([`super::Env`]),
// /// observation ([`super::Obs`]), action ([`super::Act`]) and policy ([`super::Policy`]),
// /// are interacts as illustrated in the following diagram:
// ///
// /// ```mermaid
// /// graph LR
// ///     Env --> Obs
// ///     Obs --> Policy
// ///     Policy --> Act
// ///     Act --> Env
// /// ```
// ///
// /// By definition of the environment, observations and actions can be modified.
// /// The constructor of [`crate::env::py_gym_env::PyGymEnv`] accepts
// /// [`crate::env::py_gym_env::PyGymEnvObsFilter`] and
// /// [`crate::env::py_gym_env::PyGymEnvActFilter`] for the purpose.
// /// In this case, the interaction of the entities is shown as below
// /// (`PyGymEnvAct` is for discrete or continuous actions in reality):
// ///
// /// ```mermaid
// /// graph LR
// ///     PyGymEnvObsFilter --> PyGymEnvObs
// ///     PyGymEnvObs --> Policy
// ///     Policy --> PyGymEnvAct
// ///     PyGymEnvAct --> PyGymEnvActFilter
// ///
// ///    subgraph PyGymEnv
// ///        PyGymEnvActFilter --> Py(Python runtime)
// ///        Py(Python runtime) --> PyGymEnvObsFilter
// ///    end
// /// ```
// pub fn eval<E: Env, P: Policy<E>>(env: &mut E, policy: &mut P, n_episodes: usize) -> Vec<f32> {
//     // TODO: check the maximum number of steps of the environment for evaluation.
//     // If it is infinite, the number of evaluation steps should be given in place of
//     // n_episodes.
//     let mut rs = Vec::new();
//     let obs = env.reset(None).unwrap();
//     let obs_prev = RefCell::new(Some(obs));

//     for i in 0..n_episodes {
//         let mut r_sum = 0.0;
//         let mut steps = 0;
//         loop {
//             let (step, _) = sample(env, policy, &obs_prev);
//             r_sum += &step.reward[0];
//             if step.is_done[0] == 1 {
//                 break;
//             } else {
//                 steps += 1;
//             }
//         }
//         rs.push(r_sum);
//         info!("Episode {:?}, {:?} steps, reward = {:?}", i, steps, r_sum);
//     }
//     rs
// }

/// Run episodes with a policy and recorder.
///
/// It supports non-vectorized environments.
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
        let mut prev_obs = env.reset(None)?;
        let mut count_step = 0;
        let mut r_total = 0.0;

        loop {
            let act = policy.sample(&prev_obs);
            let (step, mut record) = env.step(&act);
            r_total += step.reward[0];
            if step.is_done[0] == 1 {
                break;
            }
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
