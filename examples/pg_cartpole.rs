use std::error::Error;
use tch::nn;
use lrr::{
    core::{Trainer, Agent, util},
    py_gym_env::{PyGymEnv, PyGymEnvObs, PyGymEnvObsRawFilter},
    agents::{OptInterval,
        tch::{
            {Shape, ReplayBuffer},
            pg::discrete::PGDiscrete,
            model::Model1_1,
            py_gym_env::{
                obs::TchPyGymEnvObsBuffer,
                act_d::{TchPyGymEnvDiscreteAct, TchPyGymEnvDiscreteActRawFilter,
                    TchPyGymEnvDiscreteActBuffer}
            }
        }
    }
};

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[4]
    }
}

fn create_actor() -> Model1_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", 64, out_dim as _, Default::default()));
    Model1_1::new(4, 2, 3e-4, network_fn)
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64>;
type ActFilter = TchPyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f64>;
type Act = TchPyGymEnvDiscreteAct<ActFilter>;
type Env = PyGymEnv<Obs, Act, ObsFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer<ActFilter>;

fn create_agent() -> impl Agent<Env> {
    let actor = create_actor();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(10000, 1);
    let agent: PGDiscrete<Env, _, _, _> = PGDiscrete::new(
        actor, replay_buffer)
        .opt_interval(OptInterval::Steps(50))
        .n_updates_per_opt(1)
        .batch_size(64)
        .discount_factor(0.99);
    agent
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::new();
    Env::new("CartPole-v0", obs_filter, false).unwrap()
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
    trainer.get_agent().save("./examples/model/pg_cartpole")?;

    let mut env = create_env();
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/pg_cartpole")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
