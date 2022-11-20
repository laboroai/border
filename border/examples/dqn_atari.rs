mod util_dqn_atari;
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
use border_derive::{Act, SubBatch};
use border_py_gym_env::{
    FrameStackFilter, PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct,
    PyGymEnvDiscreteActRawFilter, PyGymEnvObs,
};
use border_tch_agent::{
    cnn::Cnn,
    dqn::{DqnConfig, Dqn as Dqn_},
    TensorSubBatch,
};
use clap::{App, Arg, ArgMatches};
use util_dqn_atari::{model_dir as model_dir_, Params};

const N_STACK: i64 = 4;

type PyObsDtype = u8;
type ObsDtype = u8;

shape!(ObsShape, [N_STACK as usize, 1, 84, 84]);
shape!(ActShape, [1]);

// #[derive(Clone, Debug, Obs)]
// struct Obs(PyGymEnvObs<ObsShape, PyObsDtype, ObsDtype>);
type Obs = PyGymEnvObs<ObsShape, PyObsDtype, ObsDtype>;

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, ObsDtype>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

// Wrap `PyGymEnvDiscreteAct` to make a new type.
// Act also implements Into<Tensor>.
// TODO: Consider to implement Into<Tensor> on PyGymEnvDiscreteAct when feature=tch.
#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvDiscreteAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, i64>);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

type ObsFilter = FrameStackFilter<ObsShape, PyObsDtype, ObsDtype, Obs>;
type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type EnvConfig = PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Dqn = Dqn_<Env, Cnn, ReplayBuffer>;

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
    let name = matches.value_of("name").unwrap();

    if matches.is_present("play") {
        matches.value_of("play").unwrap().to_string()
    } else if matches.is_present("play-gdrive") {
        if name == "PongNoFrameskip-v4" {
            let file_base = "dqn_PongNoFrameskip-v4_20210428_ec2";
            let url =
                "https://drive.google.com/uc?export=download&id=1TF5aN9fH5wd4APFHj9RP1JxuVNoi6lqJ";
            let model_dir = get_model_from_url(url, file_base).unwrap();
            model_dir.as_ref().to_str().unwrap().to_string()
        } else {
            panic!("Failed to download the model for {:?}", name);
        }
    } else {
        panic!("Failed to download the model for {:?}", name);
    }
}

fn env_config(name: &str) -> EnvConfig {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name(name.to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
        .atari_wrapper(Some(border_py_gym_env::AtariWrapper::Eval))
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
    let env_config_train =
        env_config(name).atari_wrapper(Some(border_py_gym_env::AtariWrapper::Train));
    let env_config_eval =
        env_config(name).atari_wrapper(Some(border_py_gym_env::AtariWrapper::Eval));
    let n_actions = n_actions(&env_config_train)?;

    // Configurations
    let agent_config = load_dqn_config(model_dir.as_str())?
        .out_dim(n_actions as _)
        .device(tch::Device::cuda_if_available());
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
        let mut agent = Dqn::build(agent_config);
        trainer.train(&mut agent, &mut recorder)?;
    }

    Ok(())
}

fn play(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = model_dir_for_play(&matches);
    let env_config = env_config(name);
    let n_actions = n_actions(&env_config)?;
    let agent_config = load_dqn_config(model_dir.as_str())?
        .out_dim(n_actions as _)
        .device(tch::Device::cuda_if_available());
    let mut agent = Dqn::build(agent_config);
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();

    env.set_render(true);
    agent.load(model_dir + "/best")?;
    agent.eval();

    let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;

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
