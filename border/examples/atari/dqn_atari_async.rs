mod util_dqn_atari;
use anyhow::Result;
use border_async_trainer::{
    actor_stats_fmt, ActorManager as ActorManager_, ActorManagerConfig,
    AsyncTrainer as AsyncTrainer_, AsyncTrainerConfig,
};
use border_atari_env::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_core::{
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    DefaultEvaluator, Env as _,
};
use border_derive::{Act, SubBatch};
use border_tch_agent::{
    cnn::Cnn,
    dqn::{Dqn, DqnConfig, DqnExplorer, EpsilonGreedy},
    TensorSubBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::{App, Arg, ArgMatches};
use crossbeam_channel::unbounded;
use std::{
    default::Default,
    sync::{Arc, Mutex},
};
use util_dqn_atari::{model_dir_async as model_dir_async_, Params};

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
type Agent = Dqn<Env, Cnn, ReplayBuffer>;
type ActorManager = ActorManager_<Agent, Env, ReplayBuffer, StepProc>;
type AsyncTrainer = AsyncTrainer_<Agent, Env, ReplayBuffer>;
type Evaluator = DefaultEvaluator<Env, Agent>;

fn env_config(name: impl Into<String>) -> EnvConfig {
    BorderAtariEnvConfig::default().name(name.into())
}

fn parse_args<'a>() -> ArgMatches<'a> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_atari_async")
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
            Arg::with_name("show-config")
                .long("show-config")
                .takes_value(false)
                .help("Showing configuration loaded from files"),
        )
        .arg(
            Arg::with_name("n-actors")
                .long("n-actors")
                .takes_value(true)
                .default_value("6")
                .help("The number of actors"),
        )
        .arg(
            Arg::with_name("eps-min")
                .long("eps-min")
                .takes_value(true)
                .default_value("0.001")
                .help("The minimum value of exploration noise probability"),
        )
        .arg(
            Arg::with_name("eps-max")
                .long("eps-max")
                .takes_value(true)
                .default_value("0.4")
                .help("The maximum value of exploration noise probability"),
        )
        .get_matches();

    matches
}

fn show_config(
    env_config: &EnvConfig,
    agent_config: &DqnConfig<Cnn>,
    actor_man_config: &ActorManagerConfig,
    trainer_config: &AsyncTrainerConfig,
) {
    println!("Device: {:?}", tch::Device::cuda_if_available());
    println!("{}", serde_yaml::to_string(&env_config).unwrap());
    println!("{}", serde_yaml::to_string(&agent_config).unwrap());
    println!("{}", serde_yaml::to_string(&actor_man_config).unwrap());
    println!("{}", serde_yaml::to_string(&trainer_config).unwrap());
}

fn model_dir_async(matches: &ArgMatches) -> Result<String> {
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

    let model_dir = model_dir_async_(name, &params)?;

    Ok(model_dir)
}

fn n_actions(env_config: &EnvConfig) -> Result<usize> {
    Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
}

fn load_dqn_config<'a>(model_dir: impl Into<&'a str>) -> Result<DqnConfig<Cnn>> {
    let config_path = format!("{}/agent.yaml", model_dir.into());
    DqnConfig::<Cnn>::load(config_path)
}

fn load_async_trainer_config<'a>(model_dir: impl Into<&'a str>) -> Result<AsyncTrainerConfig> {
    let config_path = format!("{}/trainer.yaml", model_dir.into());
    AsyncTrainerConfig::load(config_path)
}

fn load_replay_buffer_config<'a>(
    model_dir: impl Into<&'a str>,
) -> Result<SimpleReplayBufferConfig> {
    let config_path = format!("{}/replay_buffer.yaml", model_dir.into());
    SimpleReplayBufferConfig::load(config_path)
}

fn train(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = model_dir_async(&matches)?;
    let env_config_train = env_config(name);
    let n_actions = n_actions(&env_config_train)?;

    // exploration parameters
    let n_actors = matches
        .value_of("n-actors")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let eps_min = matches.value_of("eps-min").unwrap().parse::<f64>().unwrap();
    let eps_max = matches.value_of("eps-max").unwrap().parse::<f64>().unwrap();

    // Configurations
    let agent_config = load_dqn_config(model_dir.as_str())?
        .out_dim(n_actions as _)
        .device(tch::Device::cuda_if_available());
    let agent_configs = (0..n_actors)
        .map(|ix| {
            let n = ix as f64 / ((n_actors - 1) as f64);
            let eps = (eps_max - eps_min) * n + eps_min;
            let explorer =
                DqnExplorer::EpsilonGreedy(EpsilonGreedy::new().eps_start(eps).eps_final(eps));
            agent_config
                .clone()
                .device(tch::Device::Cpu)
                .explorer(explorer)
        })
        .collect::<Vec<_>>();
    let env_config_eval = env_config(name).eval();
    let replay_buffer_config = load_replay_buffer_config(model_dir.as_str())?;
    let step_proc_config = SimpleStepProcessorConfig::default();
    let actor_man_config = ActorManagerConfig::default();
    let async_trainer_config = load_async_trainer_config(model_dir.as_str())?;

    if matches.is_present("show-config") {
        show_config(
            &env_config_train,
            &agent_config,
            &actor_man_config,
            &async_trainer_config,
        );
    } else {
        let mut recorder = TensorboardRecorder::new(model_dir);
        let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;

        // Shared flag to stop actor threads
        let stop = Arc::new(Mutex::new(false));

        // Creates channels
        let (item_s, item_r) = unbounded(); // items pushed to replay buffer
        let (model_s, model_r) = unbounded(); // model_info

        // guard for initialization of envs in multiple threads
        let guard_init_env = Arc::new(Mutex::new(true));

        // Actor manager and async trainer
        let mut actors = ActorManager::build(
            &actor_man_config,
            &agent_configs,
            &env_config_train,
            &step_proc_config,
            item_s,
            model_r,
            stop.clone(),
        );
        let mut trainer = AsyncTrainer::build(
            &async_trainer_config,
            &agent_config,
            &env_config_eval,
            &replay_buffer_config,
            item_r,
            model_s,
            stop.clone(),
        );

        // Set the number of threads
        tch::set_num_threads(1);

        // Starts sampling and training
        actors.run(guard_init_env.clone());
        let stats = trainer.train(&mut recorder, &mut evaluator, guard_init_env);
        println!("Stats of async trainer");
        println!("{}", stats.fmt());

        let stats = actors.stop_and_join();
        println!("Stats of generated samples in actors");
        println!("{}", actor_stats_fmt(&stats));
    }

    Ok(())
}

fn main() -> Result<()> {
    let matches = parse_args();

    train(matches)?;

    Ok(())
}
