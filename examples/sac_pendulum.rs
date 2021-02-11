use std::error::Error;
use tch::nn;
use pyo3::PyObject;

use lrr::{
    core::{Trainer, Agent, util::eval, record::NullTrainRecorder},
    env::py_gym_env::{
        PyGymEnv, PyGymEnvActFilter,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_c::{PyGymEnvContinuousAct, to_pyobj},
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_c::TchPyGymEnvContinuousActBuffer
        }
    },
    agent::{
        OptInterval,
        tch::{
            SAC, ReplayBuffer, Shape,
            model::{Model1_2, Model2_1},
        }
    }
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

fn create_actor() -> Model1_2 {
    let network_fn = |p: &nn::Path, in_dim, hidden_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", 64, hidden_dim as _, Default::default()));
    Model1_2::new(3, 64, 1, 3e-4, network_fn)
}

fn create_critic() -> Model2_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim as _, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl3", 64, out_dim as _, Default::default()));
        Model2_1::new(4, 1, 3e-4, network_fn)
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f64>;
type Obs = PyGymEnvObs<ObsShape, f64>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f64>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

// Custom act filter
#[derive(Clone, Debug)]
struct ActFilter {}

impl PyGymEnvActFilter<Act> for ActFilter {
    fn filt(&mut self, act: Act) -> PyObject {
        to_pyobj::<ActShape>(2f32 * act.act)
    }
}

fn create_agent() -> impl Agent<Env> {
    let actor = create_actor();
    let critic = create_critic();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(100_000, 1);
    let agent: SAC<Env, _, _, _, _> = SAC::new(
        critic,
        actor,
        replay_buffer)
        .opt_interval(OptInterval::Steps(1))
        .n_updates_per_opt(1)
        .min_transitions_warmup(1000)
        .batch_size(128)
        .discount_factor(0.99)
        .tau(0.001)
        .alpha(1.0);
    agent
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default(); //new();
    let act_filter = ActFilter {}; //new();
    Env::new("Pendulum-v0", obs_filter, act_filter)
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
        .eval_interval(10_000)
        .n_episodes_per_eval(5);
    let mut recorder = NullTrainRecorder {};

    trainer.train(&mut recorder);
    trainer.get_agent().save("./examples/model/sac_pendulum")?;

    let mut env = create_env();
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/sac_pendulum")?;
    agent.eval();
    eval(&mut env, &mut agent, 5);

    Ok(())
}
