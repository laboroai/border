use std::error::Error;
use lrr::core::{Trainer, Policy, Agent, util};
use lrr::py_gym_env::{PyGymEnv, PyGymDiscreteAct};
use lrr::py_gym_env::adapter::{PyNDArrayObsAdapter, PyGymDiscreteActAdapter};
use lrr::agents::{DQN, dqn::QNetwork, ReplayBuffer};

fn create_agent() -> impl Agent<PyGymEnv<PyGymDiscreteAct>> {
    let qnet = QNetwork::new(4, 2, 0.001);
    let from_obs = PyNDArrayObsAdapter::new(&[4]);
    let into_act = PyGymDiscreteActAdapter::new(&[1]);
    let replay_buffer = ReplayBuffer::<PyGymEnv<PyGymDiscreteAct>, _, _>::new(10000, &from_obs, &into_act);
    let agent: DQN<PyGymEnv<PyGymDiscreteAct>, _, _, _> = DQN::new(
        qnet,
        replay_buffer,
        from_obs,
        into_act)
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
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    tch::manual_seed(42);

    let env = PyGymEnv::<PyGymDiscreteAct>::new("CartPole-v0")?;
    let env_eval = PyGymEnv::<PyGymDiscreteAct>::new("CartPole-v0")?;
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(1000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/test5")?;

    let mut env = PyGymEnv::<PyGymDiscreteAct>::new("CartPole-v0")?;
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/test5")?;
    agent.eval();
    util::eval(&env, &agent, 5, None);

    Ok(())
}
