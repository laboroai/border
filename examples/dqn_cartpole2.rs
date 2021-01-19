use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::OptInterval;
use lrr::agents::tch::{Shape, DQN, QNetwork, ReplayBuffer};
use lrr::agents::tch::py_gym_env::{TchPyGymEnvObs, TchPyGymEnvDiscreteAct,
    TchPyGymEnvDiscreteActBuffer, TchPyGymEnvObsBuffer};

#[derive(Debug, Clone)]
struct CartPoleObsShape {}

impl Shape for CartPoleObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

type CartPoleEnv = PyGymEnv<TchPyGymEnvObs<CartPoleObsShape, f64>, TchPyGymEnvDiscreteAct>;

fn create_agent() -> impl Agent<CartPoleEnv> {
    type E = CartPoleEnv;
    type O = TchPyGymEnvObsBuffer<CartPoleObsShape, f64>;
    type A = TchPyGymEnvDiscreteActBuffer;

    let qnet = QNetwork::new(4, 2, 0.001);
    let replay_buffer 
        = ReplayBuffer::<E, O, A>::new(10000, 1);
    let agent: DQN<E, _, _, _> = DQN::new(
        qnet,
        replay_buffer)
        .opt_interval(OptInterval::Steps(50))
        .n_updates_per_opt(1)
        .n_opts_per_soft_update(1)
        .min_transitions_warmup(100)
        .batch_size(64)
        .discount_factor(0.99)
        .tau(0.005);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    // std::env::set_var("RUST_LOG", "info");
    // std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    tch::manual_seed(42);

    let env = CartPoleEnv::new("CartPole-v0", false)?;
    let env_eval = CartPoleEnv::new("CartPole-v0", false)?;
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

    let mut env = CartPoleEnv::new("CartPole-v0", false)?;
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/dqn_cartpole")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
