use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::{Shape, DQN, QNetwork, ReplayBuffer};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvDiscreteAct,
    TchPyGymEnvDiscreteActBuffer, TchPyGymEnvObsBuffer};
use lrr::agents::tch::model::Model1_1;

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

    // let qnet = QNetwork::new(2, 3, 0.001);
    let qnet = Model1_1::new(2, 3, 0.0001);
    let replay_buffer 
        = ReplayBuffer::<E, O, A>::new(10000, 1);
    let agent: DQN<E, _, _, _> = DQN::new(
        qnet,
        replay_buffer)
        .n_samples_per_opt(200)
        .n_updates_per_opt(10)
        .n_opts_per_soft_update(1)
        .min_transitions_warmup(1000)
        .batch_size(256)
        .discount_factor(0.99)
        .tau(0.005);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    // std::env::set_var("RUST_LOG", "info");
    // std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    tch::manual_seed(42);

    let env = MountainCarEnv::new("MountainCar-v0", false)?;
    let env_eval = MountainCarEnv::new("MountainCar-v0", false)?;
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(10000)
        .eval_interval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/dqn_mountaincar")?;

    let mut env = MountainCarEnv::new("MountainCar-v0", false)?;
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/dqn_mountaincar")?;
    agent.eval();
    util::eval(&env, &agent, 5, None);

    Ok(())
}
