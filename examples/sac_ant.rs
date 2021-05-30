use anyhow::Result;
use border::{
    agent::{
        tch::{
            sac::EntCoefMode,
            util::{create_actor, create_critic},
            ReplayBuffer, SACBuilder,
        },
        CriticLoss, OptInterval,
    },
    env::py_gym_env::{
        act_c::{PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter},
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        tch::{act_c::TchPyGymEnvContinuousActBuffer, obs::TchPyGymEnvObsBuffer},
        PyGymEnv, PyGymEnvBuilder, Shape,
    },
    util::url::get_model_from_url,
};
use border_core::{record::TensorboardRecorder, util, Agent, TrainerBuilder};
use clap::{App, Arg};
use log::info;
use std::{default::Default, time::Duration};

const OBS_DIM: i64 = 28;
const ACT_DIM: i64 = 8;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_CRITICS: usize = 2;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 10_000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.02;
// const ALPHA: f64 = 1.0;
const TARGET_ENTROPY: f64 = -(ACT_DIM as f64);
const LR_ENT_COEF: f64 = 3e-4;
const REWARD_SCALE: f32 = 1.0;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
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
        &[OBS_DIM as _]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[ACT_DIM as _]
    }

    fn squeeze_first_dim() -> bool {
        true
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f32, f32>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

fn create_agent() -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(OBS_DIM, ACT_DIM, LR_ACTOR, vec![400, 300], device)?;
    let critics = (0..N_CRITICS)
        .map(|_| {
            create_critic(OBS_DIM + ACT_DIM, 1, LR_CRITIC, vec![400, 300], device)
                .expect("Failed to create critic")
        })
        .collect::<Vec<_>>();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, 1);

    Ok(SACBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .reward_scale(REWARD_SCALE)
        .critic_loss(CRITIC_LOSS)
        .build(critics, actor, replay_buffer, device))
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .pybullet(true)
        .atari_wrapper(None)
        .build("AntPyBulletEnv-v0", obs_filter, act_filter)
        .unwrap()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("play")
                .long("play")
                .takes_value(true)
                .help("Play with the trained model of the given path"),
        )
        .arg(
            Arg::with_name("play-gdrive")
                .long("play-gdrive")
                .takes_value(false)
                .help("Play with the trained model downloaded from google drive"),
        )
        .arg(
            Arg::with_name("wait")
                .long("wait")
                .takes_value(true)
                .default_value("25")
                .help("Waiting time in milliseconds between frames when playing"),
        )
        .get_matches();

    if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent()?;
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new("./examples/model/sac_ant");

        trainer.train(&mut recorder);
        trainer
            .get_agent()
            .save("./examples/model/sac_ant")
            .unwrap();
    } else {
        let mut env = create_env();
        let mut agent = create_agent()?;

        if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        } else {
            let file_base = "sac_ant_20210324_ec2_smoothl1";
            let url =
                "https://drive.google.com/uc?export=download&id=1XvFi2nJD5OhpTvs-Et3YREuoqy8c3Vkq";
            let model_dir = get_model_from_url(url, file_base)?;
            info!("Download the model in {:?}", model_dir.as_ref().to_str());
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        };

        let time = matches.value_of("wait").unwrap().parse::<u64>()?;
        env.set_render(true);
        env.set_wait_in_render(Duration::from_millis(time));
        agent.eval();
        util::eval(&mut env, &mut agent, 5);
    }

    Ok(())
}
