use std::error::Error;
use lrr::core::{Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::OptInterval;
use lrr::agents::tch::{DQN, ReplayBuffer};
use lrr::agents::tch::py_gym_env::{
    TchPyGymEnvObs, TchPyGymEnvObsBuffer,
    TchPyGymEnvDiscreteAct, TchPyGymEnvDiscreteActBuffer
};
use lrr::agents::tch::py_gym_env::pong::{
    PongObsShape, PongObsFilter, PongNet, PongActFilter
};

type ObsShape = PongObsShape;
type ObsFilter = PongObsFilter;
type ActFilter = PongActFilter;
type Obs = TchPyGymEnvObs<ObsShape, u8>;
type Act = TchPyGymEnvDiscreteAct<ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer<ActFilter>;
type Env = PyGymEnv<Obs, Act, ObsFilter>;

fn create_agent() -> impl Agent<Env> {
    let qnet = PongNet::new(0.001);
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(10000, 1);
    let agent: DQN<Env, _, _, _> = DQN::new(
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

fn create_env() -> Env {
    let obs_filter = ObsFilter::new();
    Env::new("Pong-v0", obs_filter, false).unwrap()
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
        .max_opts(10000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/dqn_pong")?;

    let mut env = create_env();
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/dqn_pong")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
