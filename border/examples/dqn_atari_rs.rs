mod util_dqn_atari;
use anyhow::Result;
use border_atari_env::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_core::{
    record::{BufferedRecorder, TensorboardRecorder},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    util, Policy, Agent, Env as _, Trainer, TrainerConfig,
};
use border_derive::{Act, SubBatch};
use border_tch_agent::{
    cnn::Cnn,
    dqn::{DqnConfig, Dqn as Dqn_},
    TensorSubBatch,
};
use clap::{App, Arg, ArgMatches};
use util_dqn_atari::{model_dir as model_dir_, Params};

type ObsDtype = u8;

// #[derive(Debug, Clone, Obs)]
// struct Obs(BorderAtariObs);
type Obs = BorderAtariObs;

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(SubBatch)]
// struct ActBatch(TensorSubBatch<ActShape, i64>);
struct ActBatch(TensorSubBatch);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

// Wrap `BorderAtariAct` to make a new type.
// Act also implements Into<Tensor>.
// TODO: Consider to implement Into<Tensor> on BorderAtariAct when feature=tch.
#[derive(Debug, Clone, Act)]
struct Act(BorderAtariAct);

type ObsFilter = BorderAtariObsRawFilter<Obs>;
type ActFilter = BorderAtariActRawFilter<Act>;
type EnvConfig = BorderAtariEnvConfig<Obs, Act, ObsFilter, ActFilter>;
type Env = BorderAtariEnv<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Dqn = Dqn_<Env, Cnn, ReplayBuffer>;

fn env_config(name: impl Into<String>) -> EnvConfig {
    BorderAtariEnvConfig::default().name(name.into())
}

fn init<'a>() -> ArgMatches<'a> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_atari")
        .version("0.1.0")
        .author("Taku Yoshioka <yoshioka@laboro.ai>")
        .arg(
            Arg::with_name("name")
                .long("name")
                .takes_value(true)
                .required(true)
                .index(1)
                .help("The name of the atari environment (e.g., PongNoFrameskip-v4)"),
        )
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
            Arg::with_name("per")
                .long("per")
                .takes_value(false)
                .help("Train/play with prioritized experience replay"),
        )
        .arg(
            Arg::with_name("ddqn")
                .long("ddqn")
                .takes_value(false)
                .help("Train/play with double DQN"),
        )
        .arg(
            Arg::with_name("debug")
                .long("debug")
                .takes_value(false)
                .help("Run with debug configuration"),
        )
        .arg(
            Arg::with_name("wait")
                .long("wait")
                .takes_value(true)
                .default_value("25")
                .help("Waiting time in milliseconds between frames when playing"),
        )
        .arg(
            Arg::with_name("show-config")
                .long("show-config")
                .takes_value(false)
                .help("Showing configuration loaded from files"),
        )
        .get_matches();

    matches
}

fn show_config(
    env_config: &EnvConfig,
    agent_config: &DqnConfig<Cnn>,
    trainer_config: &TrainerConfig,
) {
    println!("Device: {:?}", tch::Device::cuda_if_available());
    println!("{}", serde_yaml::to_string(&env_config).unwrap());
    println!("{}", serde_yaml::to_string(&agent_config).unwrap());
    println!("{}", serde_yaml::to_string(&trainer_config).unwrap());
}

fn model_dir(matches: &ArgMatches) -> Result<String> {
    let name = matches
        .value_of("name")
        .expect("The name of the environment was not given")
        .to_string();
    let mut params = Params::default();

    if matches.is_present("ddqn") {
        params = params.ddqn();
    }

    if matches.is_present("per") {
        params = params.per();
    }

    if matches.is_present("debug") {
        params = params.debug();
    }

    model_dir_(name, &params)
}

fn model_dir_for_play(matches: &ArgMatches) -> String {
    matches.value_of("play").unwrap().to_string()
}

fn n_actions(env_config: &EnvConfig) -> Result<usize> {
    Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
}

fn load_dqn_config<'a>(model_dir: impl Into<&'a str>) -> Result<DqnConfig<Cnn>> {
    let config_path = format!("{}/agent.yaml", model_dir.into());
    DqnConfig::<Cnn>::load(config_path)
}

fn load_trainer_config<'a>(model_dir: impl Into<&'a str>) -> Result<TrainerConfig> {
    let config_path = format!("{}/trainer.yaml", model_dir.into());
    TrainerConfig::load(config_path)
}

fn load_replay_buffer_config<'a>(
    model_dir: impl Into<&'a str>,
) -> Result<SimpleReplayBufferConfig> {
    let config_path = format!("{}/replay_buffer.yaml", model_dir.into());
    SimpleReplayBufferConfig::load(config_path)
}

fn train(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = model_dir(&matches)?;
    let env_config_train = env_config(name);
    let env_config_eval = env_config(name).eval();
    let n_actions = n_actions(&env_config_train)?;

    // Configurations
    let agent_config = load_dqn_config(model_dir.as_str())?.out_dim(n_actions as _);
    let trainer_config = load_trainer_config(model_dir.as_str())?;
    let replay_buffer_config = load_replay_buffer_config(model_dir.as_str())?;
    let step_proc_config = SimpleStepProcessorConfig {};

    if matches.is_present("show-config") {
        show_config(&env_config_train, &agent_config, &trainer_config);
    } else {
        let mut trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            trainer_config,
            env_config_train,
            Some(env_config_eval),
            step_proc_config,
            replay_buffer_config,
        );
        let mut recorder = TensorboardRecorder::new(model_dir);
        let agent_config = agent_config.device(tch::Device::cuda_if_available());
        let mut agent = Dqn::build(agent_config);
        trainer.train(&mut agent, &mut recorder)?;
    }

    Ok(())
}

fn play(matches: ArgMatches) -> Result<()> {
    let device = tch::Device::cuda_if_available();
    let name = matches.value_of("name").unwrap();
    let model_dir = model_dir_for_play(&matches);
    let env_config = env_config(name);
    let n_actions = n_actions(&env_config)?;
    let agent_config = load_dqn_config(model_dir.as_str())?
        .out_dim(n_actions as _)
        .device(device);
    let mut agent = Dqn::build(agent_config);
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();

    env.open()?;
    agent.load(model_dir + "/best")?;
    agent.eval();

    let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;

    // env.close();

    Ok(())
}

fn main() -> Result<()> {
    let matches = init();

    if matches.is_present("play") || matches.is_present("play-gdrive") {
        play(matches)?;
    } else {
        train(matches)?;
    }

    Ok(())
}
