use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::OptInterval;
use lrr::agents::tch::{SAC, ReplayBuffer, Shape};
use lrr::agents::tch::model::{Model1_2, Model2_1};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvContinuousAct,
    TchPyGymEnvContinuousActBuffer, TchPyGymEnvObsBuffer};
use lrr::agents::tch::py_gym_env::act_c::RawFilter;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[8]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[2]
    }

    fn squeeze_first_dim() -> bool {
        true
    }
}

type E = PyGymEnv<TchPyGymEnvObs<ObsShape, f32>, TchPyGymEnvContinuousAct<ActShape, RawFilter>>;
type O = TchPyGymEnvObsBuffer<ObsShape, f32>;
type A = TchPyGymEnvContinuousActBuffer<ActShape, RawFilter>;

fn create_agent() -> impl Agent<E> {
    let qnet = Model2_1::new(10, 1, 1e-3);
    let pi = Model1_2::new(8, 2, 1e-4);
    let replay_buffer
        = ReplayBuffer::<E, O, A>::new(100_000, 1);
    let agent: SAC<E, _, _, _, _> = SAC::new(
        qnet,
        pi,
        replay_buffer)
        .opt_interval(OptInterval::Steps(200))
        .n_updates_per_opt(200)
        .min_transitions_warmup(200)
        .batch_size(100)
        .discount_factor(0.99)
        .tau(0.5)
        .alpha(1.0);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);

    let env = E::new("LunarLanderContinuous-v2", true)?.max_steps(None);
    let env_eval = E::new("LunarLanderContinuous-v2", true)?.max_steps(None);
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(100)
        .n_opts_per_eval(1)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/sac_lunarlander_cont")?;

    let mut env = E::new("LunarLanderContinuous-v2", true)?.max_steps(None);
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/sac_lunarlander_cont")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
