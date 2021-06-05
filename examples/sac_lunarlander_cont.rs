use anyhow::Result;
use border::{
    agent::{
        tch::{
            sac::EntCoefMode,
            util::mlp::{create_actor, create_critic},
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
    shape,
};
use border_core::{
    record::{BufferedRecorder, Record, TensorboardRecorder},
    util::eval_with_recorder,
    Agent, TrainerBuilder,
};
use clap::{App, Arg};
use serde::Serialize;
use std::{convert::TryFrom, fs::File};

const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_CRITICS: usize = 1;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 128;
const N_TRANSITIONS_WARMUP: usize = 1000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.001;
const ALPHA: f64 = 0.5;
// const TARGET_ENTROPY: f64 = -(ACT_DIM as f64);
// const LR_ENT_COEF: f64 = 3e-4;
const REWARD_SCALE: f32 = 1.0;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 200_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const MAX_STEPS_IN_EPISODE: usize = 1000;
const MODEL_DIR: &str = "./examples/model/sac_lunarlander_cont";

shape!(ObsShape, [8]);
shape!(ActShape, [2], squeeze_first_dim);

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f32, f32>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

fn create_agent() -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(
        ObsShape::shape()[0] as _,
        ActShape::shape()[0] as _,
        LR_ACTOR,
        vec![64, 64],
        device,
    )?;
    let critics = (0..N_CRITICS)
        .map(|_| {
            create_critic(
                (ObsShape::shape()[0] + ActShape::shape()[0]) as _,
                1,
                LR_CRITIC,
                vec![64, 64],
                device,
            )
            .expect("Cannot create critic")
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
        // .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .ent_coef_mode(EntCoefMode::Fix(ALPHA))
        .reward_scale(REWARD_SCALE)
        .critic_loss(CRITIC_LOSS)
        .build(critics, actor, replay_buffer, device))
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .build("LunarLanderContinuous-v2", obs_filter, act_filter)
        .unwrap()
        .max_steps(Some(MAX_STEPS_IN_EPISODE)) // TODO: consider moving the method to the builder
}

#[derive(Debug, Serialize)]
struct LunarlanderRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f32>,
    act: Vec<f32>,
}

impl TryFrom<&Record> for LunarlanderRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record.get_array1("obs")?.to_vec(),
            act: record.get_array1("act")?.to_vec(),
        })
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let matches = App::new("sac_lunarlander_cont")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("skip training")
                .long("skip_training")
                .takes_value(false)
                .help("Skip training"),
        )
        .get_matches();

    if !matches.is_present("skip training") {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent()?;
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(MODEL_DIR)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(MODEL_DIR);

        trainer.train(&mut recorder);
    }

    let mut env = create_env();
    let mut agent = create_agent()?;
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load(MODEL_DIR).unwrap();
    agent.eval();

    eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("examples/model/sac_lunarlander_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(LunarlanderRecord::try_from(record)?)?;
    }

    Ok(())
}
