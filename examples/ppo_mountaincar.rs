use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::{Shape, PPODiscrete, ReplayBuffer};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvDiscreteAct,
    TchPyGymEnvDiscreteActBuffer, TchPyGymEnvObsBuffer};
use lrr::agents::tch::model::StateValueAndDiscreteActProb;

#[derive(Debug, Clone)]
struct MountainCarObsShape {}

impl Shape for MountainCarObsShape {
    fn shape() -> &'static [usize] {
        &[2]
    }
}

type MountainCarEnv = PyGymEnv<TchPyGymEnvObs<MountainCarObsShape>, TchPyGymEnvDiscreteAct>;

fn create_agent() -> impl Agent<MountainCarEnv> {
    type E = MountainCarEnv;
    type O = TchPyGymEnvObsBuffer<MountainCarObsShape>;
    type A = TchPyGymEnvDiscreteActBuffer;

    let mh_model = StateValueAndDiscreteActProb::new(2, 3, 0.0001);
    let agent: PPODiscrete<E, _, O, A> = PPODiscrete::new(
        mh_model, 100, 1)
        .n_updates_per_opt(1)
        .batch_size(64)
        .discount_factor(0.99);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    // std::env::set_var("RUST_LOG", "info");
    // std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    tch::manual_seed(42);

    let env = MountainCarEnv::new("MountainCar-v0", false)?.max_steps(Some(200));
    let env_eval = MountainCarEnv::new("MountainCar-v0", false)?.max_steps(Some(200));
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(10000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/ppo_mountaincar")?;

    let mut env = MountainCarEnv::new("MountainCar-v0", false)?.max_steps(Some(200));
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/ppo_mountaincar")?;
    agent.eval();
    util::eval(&env, &agent, 5, None);

    Ok(())
}
