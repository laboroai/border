use std::error::Error;
use tch::nn;
use lrr::{
    core::{Trainer, Agent, util},
    py_gym_env::{PyGymEnv, PyGymEnvObs},
    agents::{
        OptInterval,
        tch::{
            {Shape, ReplayBuffer},
            pg::discrete::PGDiscrete,
            model::Model1_1,
            py_gym_env::{
                obs::TchPyGymEnvObsBuffer,
                act_d::{TchPyGymEnvDiscreteAct, TchPyGymEnvDiscreteActBuffer},
                pong::{PongObsShape, PongObsFilter, PongActFilter}
            }
        }
    }
};

type ObsShape = PongObsShape;
type ObsFilter = PongObsFilter;
type ActFilter = PongActFilter;
type Obs = PyGymEnvObs<ObsShape, u8>;
type Act = TchPyGymEnvDiscreteAct<ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer<ActFilter>;
type Env = PyGymEnv<Obs, Act, ObsFilter>;

fn create_actor() -> Model1_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add_fn(move |xs| xs.reshape(&[-1, in_dim as _]))
        .add(nn::linear(p / "al1", in_dim as _, 200, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 200, out_dim as _, Default::default()));
    Model1_1::new(80 * 80, 2, 3e-4, network_fn)
}

fn create_agent() -> impl Agent<Env> {
    let actor = create_actor();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(10000, 1)
        .nonzero_reward_as_done(true);
    let agent: PGDiscrete<Env, _, _, _> = PGDiscrete::new(actor, replay_buffer)
        .opt_interval(OptInterval::Episodes(1))
        .n_updates_per_opt(5)
        .batch_size(128)
        .discount_factor(0.99);
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
    let mut trainer = Trainer::new(env, env_eval, agent)
        .max_opts(10000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/pg_pong")?;

    let mut env = create_env();
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/pg_pong")?;
    agent.eval();
    util::eval(&env, &mut agent, 5, None);

    Ok(())
}
