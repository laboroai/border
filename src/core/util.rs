use std::cell::RefCell;
use crate::core::{Step, Env, Agent};

/// The agent take an action and apply it to the environment.
/// Then return [crate::core::base::Step] object.
pub fn sample<E: Env, A: Agent<E>>(env: &E, agent: &mut A, obs_prev: &RefCell<Option<E::Obs>>) -> Step<E::Obs, E::Act, E::Info> {
    let obs = match obs_prev.replace(None) {
        None => {
            let obs = env.reset().unwrap();
            agent.push_obs(&obs);
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
