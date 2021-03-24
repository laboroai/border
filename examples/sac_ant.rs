// use std::time::Duration;
use anyhow::Result;
use clap::{Arg, App};
use tch::nn;

use border::{
    core::{
        Agent, TrainerBuilder, util,
        record::TensorboardRecorder,
    }, 
    env::py_gym_env::{
        Shape, PyGymEnv, PyGymEnvBuilder,
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        act_c::{PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter},
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_c::TchPyGymEnvContinuousActBuffer,
        }
    },
    agent::{
        OptInterval,
        tch::{
            SACBuilder, ReplayBuffer, sac::EntCoefMode,
            model::{Model1_2, Model2_1}
        }
    }
};

const DIM_OBS: usize = 28;
const DIM_ACT: usize = 8;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_CRITICS: usize = 2;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 10_000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.02;
// const ALPHA: f64 = 1.0;
const TARGET_ENTROPY: f64 = -(DIM_ACT as f64);
const LR_ENT_COEF: f64 = 3e-4;
const REWARD_SCALE: f32 = 1.0;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const N_EPISODES_PER_EVAL: usize = 5;
// const MAX_STEPS_IN_EPISODE: usize = 200;

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

    fn squeeze_first_dim() -> bool {
        true
    }
}

fn create_actor(device: tch::Device) -> Model1_2 {
    let network_fn = |p: &nn::Path, in_dim, hidden_dim| nn::seq()
        .add(nn::linear(p / "al1", in_dim as _, 400, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "al2", 400, hidden_dim as _, Default::default()))
        .add_fn(|xs| xs.relu());
    Model1_2::new(DIM_OBS, 300, DIM_ACT, LR_ACTOR, network_fn, device)
}

fn create_critic(device: tch::Device) -> Model2_1 {
    let network_fn = |p: &nn::Path, in_dim, out_dim| nn::seq()
        .add(nn::linear(p / "cl1", in_dim as _, 400, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl2", 400, 300, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "cl3", 300, out_dim as _, Default::default()));
        Model2_1::new(DIM_OBS + DIM_ACT, 1, LR_CRITIC, network_fn, device)
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f32>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

fn create_agent() -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(device);
    let critics = (0..N_CRITICS).map(|_| create_critic(device)).collect();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, 1);

    SACBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .reward_scale(REWARD_SCALE)
        .build(critics, actor, replay_buffer, device)
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .pybullet(true)
        .atari_wrapper(false)
        .build("AntPyBulletEnv-v0", obs_filter, act_filter).unwrap()
}

fn main() -> Result<()> {
    env_logger::init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(Arg::with_name("play")
            .long("play")
            .takes_value(true)
            .help("Play with the trained model of the given path"))
        .get_matches();

    if !matches.is_present("play") {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent();
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new("./examples/model/sac_ant");
    
        trainer.train(&mut recorder);
        trainer.get_agent().save("./examples/model/sac_ant").unwrap();    
    }
    else {
        let model_file = matches.value_of("play").unwrap();
        let mut env = create_env();
        let mut agent = create_agent();

        env.set_render(true);
        // env.set_wait_in_render(Duration::from_millis(50));
        agent.load(model_file).unwrap(); // TODO: define appropriate error
        agent.eval();

        util::eval(&mut env, &mut agent, 5);
    }

    Ok(())
}
