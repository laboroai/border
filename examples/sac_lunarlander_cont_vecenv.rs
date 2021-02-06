use std::error::Error;
use tch::nn;
use lrr::{
    core::{Trainer, Agent, util},
    py_gym_env::{
        PyVecGymEnv,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_c::{PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter}
    },
    agents::{
        OptInterval,
        tch::{
            SAC, ReplayBuffer, Shape,
            model::{Model1_2, Model2_1},
            py_gym_env::{
                obs::TchPyGymEnvObsBuffer,
                act_c::TchPyGymEnvContinuousActBuffer,
            }
        }
    }
};

const N_PROCS: usize = 4;
const DIM_OBS: usize = 8;
const DIM_ACT: usize = 2;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_HIDDEN: i64 = 64;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 32;
const N_TRANSITIONS_WARMUP: usize = 250;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.005;
const ALPHA: f64 = 0.5;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 200_000;
const N_OPTS_PER_EVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 25_000;
const N_EPISODES_PER_EVAL: usize = 5;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[DIM_OBS]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[DIM_ACT]
    }

    // In vectorized environment, the first axis should be considered as the number of processes.
    fn squeeze_first_dim() -> bool {
        false
    }
}

fn create_actor() -> Model1_2 {
    let network_fn = |p: &nn::Path, in_dim, hidden_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, N_HIDDEN, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", N_HIDDEN, N_HIDDEN, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al3", N_HIDDEN, hidden_dim as _, Default::default()));
    Model1_2::new(DIM_OBS, N_HIDDEN as usize, DIM_ACT, LR_ACTOR, network_fn)
}

fn create_critic() -> Model2_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim as _, N_HIDDEN, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", N_HIDDEN, N_HIDDEN, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl3", N_HIDDEN, out_dim as _, Default::default()));
        Model2_1::new(DIM_OBS + DIM_ACT, 1, LR_CRITIC, network_fn)
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f32>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

fn create_agent() -> impl Agent<Env> {
    let actor = create_actor();
    let critic = create_critic();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, N_PROCS);
    let agent: SAC<Env, _, _, _, _> = SAC::new(
        critic,
        actor,
        replay_buffer)
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .alpha(ALPHA);
    agent
}

fn create_env(n_procs: usize) -> Env {
    let obs_filters: Vec<_> = (0..n_procs).map(|_| ObsFilter::new()).collect();
    let act_filter = ActFilter { vectorized: true };
    Env::new("LunarLanderContinuous-v2", obs_filters, act_filter).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);

    let env = create_env(N_PROCS);
    let env_eval = create_env(1);
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(MAX_OPTS)
        .n_opts_per_eval(N_OPTS_PER_EVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL);

    trainer.train();
    trainer.get_agent().save("./examples/model/sac_lunarlander_cont")?;

    // let mut env = create_env();
    // let mut agent = create_agent();
    // env.set_render(true);
    // agent.load("./examples/model/sac_lunarlander_cont")?;
    // agent.eval();
    // util::eval(&env, &mut agent, 5, None);

    Ok(())
}
