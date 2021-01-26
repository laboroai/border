use std::error::Error;
use tch::nn;
use lrr::core::{Agent, Trainer, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::OptInterval;
use lrr::agents::tch::{DDPG, ReplayBuffer, Shape};
use lrr::agents::tch::model::{Model1_1, Model2_1};
use lrr::agents::tch::py_gym_env::obs::{
    TchPyGymEnvObs, TchPyGymEnvObsBuffer, TchPyGymEnvObsRawFilter
};
use lrr::agents::tch::py_gym_env::act_c::{
    TchPyGymEnvContinuousAct, TchPyGymEnvContinuousActBuffer, RawFilter
};

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

// TODO: set action scale 2.0, as sac_pendulum

fn create_actor() -> Model1_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", 64, out_dim as _, Default::default()))
        .add_fn(|xs| xs.tanh());
    Model1_1::new(3, 1, 3e-4, network_fn)
}

fn create_critic() -> Model2_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", 64, out_dim as _, Default::default()));
        Model2_1::new(4, 1, 3e-4, network_fn)
}

type ObsFilter = TchPyGymEnvObsRawFilter<ObsShape, f64>;
type ActFilter = RawFilter;
type Obs = TchPyGymEnvObs<ObsShape, f64>;
type Act = TchPyGymEnvContinuousAct<ActShape, ActFilter>;
type Env = PyGymEnv<Obs, Act, ObsFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape, ActFilter>;

fn create_agent() -> impl Agent<Env> {
    let actor = create_actor();
    let critic = create_critic();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(100_000, 1);
    let agent: DDPG<Env, _, _, _, _> = DDPG::new(
        critic,
        actor,
        replay_buffer)
        .opt_interval(OptInterval::Steps(1))
        .n_updates_per_opt(1)
        .min_transitions_warmup(1000)
        .batch_size(128)
        .discount_factor(0.99)
        .tau(0.001);
    agent
}

fn create_env() -> Env {
    let obs_filter = TchPyGymEnvObsRawFilter::new();
    Env::new("Pendulum-v0", obs_filter, true)
        .unwrap()
        .max_steps(Some(200))
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
        .max_opts(200_000)
        .n_opts_per_eval(10_000)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/ddpg_pendulum")?;

    let mut env = create_env();
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/ddpg_pendulum")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
