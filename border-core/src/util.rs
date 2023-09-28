#![allow(clippy::float_cmp)]
//! Utilities for interaction of agents and environments.
use crate::{
    record::{RecordValue, Recorder},
    Env, Policy
};
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

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

/// Runs episodes with a policy and recorder.
///
/// This function does not support vectorized environments.
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

pub fn shuffle<T>(mut vec: Vec<T>, seed: [u8; 32]) -> Vec<T> {
    let mut rng = StdRng::from_seed(seed);
    vec.shuffle(&mut rng);
    vec
}
