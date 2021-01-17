use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::{DDPG, ReplayBuffer, Shape};
use lrr::agents::tch::model::{Model1_1, Model2_1};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvContinuousAct,
    TchPyGymEnvContinuousActBuffer, TchPyGymEnvObsBuffer};

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[3]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[1]
    }

    fn squeeze_first_dim() -> bool {
        true
    }
}

type E = PyGymEnv<TchPyGymEnvObs<ObsShape>, TchPyGymEnvContinuousAct<ActShape>>;
type O = TchPyGymEnvObsBuffer<ObsShape>;
type A = TchPyGymEnvContinuousActBuffer<ActShape>;

fn create_agent() -> impl Agent<E> {
    let critic = Model2_1::new(4, 1, 1e-3);
    let actor = Model1_1::new(3, 1, 1e-4);
    let replay_buffer
        = ReplayBuffer::<E, O, A>::new(100_000, 1);
    let agent: DDPG<E, _, _, _, _> = DDPG::new(
        critic,
        actor,
        replay_buffer)
        .n_samples_per_opt(200)
        .n_updates_per_opt(200)
        .min_transitions_warmup(200)
        .batch_size(100)
        .discount_factor(0.99)
        .n_opts_per_soft_update(1)
        .tau(0.005);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    // std::env::set_var("RUST_LOG", "trace");
    // std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    tch::manual_seed(42);

    let env = E::new("Pendulum-v0", true)?.max_steps(Some(200));
    let env_eval = E::new("Pendulum-v0", true)?.max_steps(Some(200));
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(200 * 1000)
        .n_opts_per_eval(1)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/ddpg_pendulum")?;

    let mut env = E::new("Pendulum-v0", true)?.max_steps(Some(200));
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/ddpg_pendulum")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
