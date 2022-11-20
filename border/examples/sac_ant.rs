use anyhow::Result;
use border::util::get_model_from_url;
use border_core::{
    record::{BufferedRecorder, TensorboardRecorder},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    shape, util, Agent, Env as _, Policy, Trainer, TrainerConfig,
};
use border_derive::{Act, Obs, SubBatch};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvContinuousAct,
    PyGymEnvContinuousActRawFilter, PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};
use border_tch_agent::{
    mlp::{MlpConfig, Mlp, Mlp2},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, EntCoefMode, SacConfig, Sac},
    util::CriticLoss,
    TensorSubBatch,
};
use clap::{App, Arg};
use log::info;
use std::{convert::TryFrom, time::Duration};

const DIM_OBS: i64 = 28;
const DIM_ACT: i64 = 8;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 10_000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const N_EPISODES_PER_EVAL: usize = 5;
const N_CRITICS: usize = 2;
const TAU: f64 = 0.02;
const TARGET_ENTROPY: f64 = -(DIM_ACT as f64);
const LR_ENT_COEF: f64 = 3e-4;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const MODEL_DIR: &str = "./border/examples/model/sac_ant";

type PyObsDtype = f32;

shape!(ObsShape, [DIM_OBS as _]);
shape!(ActShape, [DIM_ACT as _]);

#[derive(Clone, Debug, Obs)]
struct Obs(PyGymEnvObs<ObsShape, PyObsDtype, f32>);

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, f32>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvContinuousAct<ActShape>);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, f32>);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, PyObsDtype, f32, Obs>;
type ActFilter = PyGymEnvContinuousActRawFilter<ActShape, Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;

fn create_agent(in_dim: i64, out_dim: i64) -> Sac<Env, Mlp, Mlp2, ReplayBuffer> {
    let device = tch::Device::cuda_if_available();
    let actor_config = ActorConfig::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
        .out_dim(out_dim)
        .pi_config(MlpConfig::new(in_dim, vec![400, 300], out_dim, true));
    let critic_config = CriticConfig::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
        .q_config(MlpConfig::new(in_dim + out_dim, vec![400, 300], 1, true));
    let sac_config = SacConfig::default()
        .batch_size(BATCH_SIZE)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .actor_config(actor_config)
        .critic_config(critic_config)
        .tau(TAU)
        .critic_loss(CRITIC_LOSS)
        .n_critics(N_CRITICS)
        .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .device(device);
    Sac::build(sac_config)
}

fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("AntPyBulletEnv-v0".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
        .pybullet(true)
}

fn train(max_opts: usize, model_dir: &str) -> Result<()> {
    let mut trainer = {
        let env_config = env_config();
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
        let config = TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(EVAL_INTERVAL)
            .record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .eval_episodes(N_EPISODES_PER_EVAL)
            .model_dir(model_dir);
        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            config,
            env_config,
            None,
            step_proc_config,
            replay_buffer_config,
        );

        trainer
    };

    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut agent = create_agent(DIM_OBS, DIM_ACT);

    trainer.train(&mut agent, &mut recorder)?;

    Ok(())
}

fn eval(model_dir: &str) -> Result<()> {
    let mut env = Env::build(&env_config(), 0)?;
    env.set_wait_in_render(Duration::from_millis(10));
    let mut agent = create_agent(DIM_OBS, DIM_ACT);
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load(model_dir)?;
    agent.eval();

    let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <yoshioka@laboro.ai>")
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
        train(MAX_OPTS, MODEL_DIR)?;
    } else {
        let model_dir = if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            format!("{}{}", model_dir, "/best").to_owned()
        } else {
            let file_base = "sac_ant_20210324_ec2_smoothl1";
            let url =
                "https://drive.google.com/uc?export=download&id=1XvFi2nJD5OhpTvs-Et3YREuoqy8c3Vkq";
            let model_dir = get_model_from_url(url, file_base)?;
            info!("Download the model in {:?}", model_dir.as_ref().to_str());
            model_dir.as_ref().to_str().unwrap().to_string()
        };

        eval(model_dir.as_str())?;
    }

    Ok(())
}
