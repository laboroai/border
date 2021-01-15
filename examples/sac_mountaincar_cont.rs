use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::{SAC, ReplayBuffer, Shape};
use lrr::agents::tch::model::{Model1_2, Model2_1};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvContinuousAct,
    TchPyGymEnvContinuousActBuffer, TchPyGymEnvObsBuffer};

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[2]
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
    let qnet = Model2_1::new(3, 1, 1e-4);
    let pi = Model1_2::new(2, 1, 1e-4);
    let replay_buffer
        = ReplayBuffer::<E, O, A>::new(10000, 1);
    let agent: SAC<E, _, _, _, _> = SAC::new(
        qnet,
        pi,
        replay_buffer)
        .n_samples_per_opt(50)
        .n_updates_per_opt(1)
        .n_opts_per_soft_update(1)
        .min_transitions_warmup(100)
        .batch_size(64)
        .discount_factor(0.99)
        .tau(0.005);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    // std::env::set_var("RUST_LOG", "trace");
    // std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    tch::manual_seed(42);

    let env = E::new("MountainCarContinuous-v0", true)?;
    let env_eval = E::new("MountainCarContinuous-v0", true)?;
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(1000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/sac_mountaincar_cont")?;

    let mut env = E::new("MountainCarContinuous-v0", true)?;
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/sac_mountaincar_cont")?;
    agent.eval();
    util::eval(&env, &agent, 5, None);

    Ok(())
}
