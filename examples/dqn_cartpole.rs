use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::OptInterval;
use lrr::agents::tch::{Shape, DQN, QNetwork, ReplayBuffer};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvDiscreteAct,
    TchPyGymEnvDiscreteActBuffer, TchPyGymEnvObsBuffer};
use lrr::agents::tch::py_gym_env::obs::TchPyGymEnvObsRawFilter;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

type E = PyGymEnv<
    TchPyGymEnvObs<ObsShape, f64>,
    TchPyGymEnvDiscreteAct,
    TchPyGymEnvObsRawFilter<ObsShape, f64>>;
type O = TchPyGymEnvObsBuffer<ObsShape, f64>;
type A = TchPyGymEnvDiscreteActBuffer;

fn create_agent() -> impl Agent<E> {
    let qnet = QNetwork::new(4, 2, 0.001);
    let replay_buffer 
        = ReplayBuffer::<E, O, A>::new(10000, 1);
    let agent: DQN<E, _, _, _> = DQN::new(
        qnet,
        replay_buffer)
        .opt_interval(OptInterval::Steps(50))
        .n_updates_per_opt(1)
        .min_transitions_warmup(100)
        .batch_size(64)
        .discount_factor(0.99)
        .tau(0.005);
    agent
}

fn create_env() -> E {
    let obs_filter = TchPyGymEnvObsRawFilter::new();
    E::new("CartPole-v0", obs_filter, false).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);

    let env = create_env();
    let env_eval = create_env();
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(1000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/dqn_cartpole")?;

    let mut env = create_env();
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/dqn_cartpole")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
