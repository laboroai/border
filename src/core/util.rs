use std::cell::RefCell;
use log::info;
use crate::core::{Step, Env, Policy, Agent};

/// The agent take an action and apply it to the environment.
/// Then return [crate::core::base::Step] object.
pub fn sample<E: Env, A: Agent<E>>(env: &E, agent: &mut A, obs_prev: &RefCell<Option<E::Obs>>) -> Step<E> {
    let obs = match obs_prev.replace(None) {
        None => {
            let obs = env.reset().unwrap();
            if agent.is_train() {
                agent.push_obs(&obs);
            }
            obs
        },
        Some(obs) => obs
    };
    let a = agent.sample(&obs);
    let step = env.step(&a);

    if step.is_done {
        obs_prev.replace(None);
    }
    else {
        obs_prev.replace(Some(step.obs.clone()));
    }

    step
}

/// The agent take an action and apply it to the environment.
/// Then return [crate::core::base::Step] object.
pub fn sample_with_policy<E: Env, P: Policy<E>>(env: &E, policy: &P, obs_prev: &RefCell<Option<E::Obs>>) -> Step<E> {
    let obs = match obs_prev.replace(None) {
        None => env.reset().unwrap(),
        Some(obs) => obs
    };
    let a = policy.sample(&obs);
    let step = env.step(&a);

    if step.is_done {
        obs_prev.replace(None);
    }
    else {
        obs_prev.replace(Some(step.obs.clone()));
    }

    step
}

pub fn eval<E: Env, P: Policy<E>>(env: &E, policy: &P, n_episodes_per_eval: usize, count_opts: Option<usize>) {
    // TODO: check the maximum number of steps of the environment for evaluation.
    // If it is infinite, the number of evaluation steps should be given in place of
    // n_episodes_per_eval.
    let mut rs = Vec::new();

    for _ in 0..n_episodes_per_eval {
        let mut r_sum = 0.0;
        let obs_prev = RefCell::new(None);
        loop {
            let step = sample_with_policy(env, policy, &obs_prev);
            r_sum += step.reward;
            if step.is_done { break; }
        }
        rs.push(r_sum);
    }

    let mean: f32 = rs.iter().sum::<f32>() / n_episodes_per_eval as f32;
    let min = rs.iter().fold(f32::NAN, |m, v| v.min(m));
    let max = rs.iter().fold(f32::NAN, |m, v| v.max(m));
    if let Some(c) = count_opts {
        info!("Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
            c, mean, min, max);
    }
    else {
        info!("Eval (mean, min, max) of r_sum: {}, {}, {}", mean, min, max);
    }
}