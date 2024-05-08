mod util_dqn_atari;
use anyhow::Result;
use border_async_trainer::{
    util::train_async, /*ActorManager as ActorManager_,*/ ActorManagerConfig,
    /*AsyncTrainer as AsyncTrainer_,*/ AsyncTrainerConfig,
};
use border_atari_env::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_core::{
    record::AggregateRecorder,
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    DefaultEvaluator, Env as _,
};
use border_derive::{Act, SubBatch};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tch_agent::{
    cnn::Cnn,
    dqn::{Dqn, DqnConfig, DqnExplorer, EpsilonGreedy},
    TensorSubBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::{App, Arg, ArgMatches};

mod obs_act_types {
    use super::*;

    pub type Obs = BorderAtariObs;

    #[derive(Clone, SubBatch)]
    pub struct ObsBatch(TensorSubBatch);

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorSubBatch::from_tensor(tensor))
        }
    }

    #[derive(SubBatch)]
    pub struct ActBatch(TensorSubBatch);

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
    pub struct Act(BorderAtariAct);

    pub type ObsFilter = BorderAtariObsRawFilter<Obs>;
    pub type ActFilter = BorderAtariActRawFilter<Act>;
    pub type EnvConfig = BorderAtariEnvConfig<Obs, Act, ObsFilter, ActFilter>;
    pub type Env = BorderAtariEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Agent = Dqn<Env, Cnn, ReplayBuffer>;
    pub type Evaluator = DefaultEvaluator<Env, Agent>;
}

use config::DqnAtariAsyncConfig;
use obs_act_types::*;

mod config {
    use self::util_dqn_atari::{
        DqnAtariAgentConfig, DqnAtariAsyncTrainerConfig, DqnAtariReplayBufferConfig,
    };
    use serde::Serialize;
    use std::io::Write;

    use super::*;

    pub fn env_config(name: impl Into<String>) -> EnvConfig {
        BorderAtariEnvConfig::default().name(name.into())
    }

    pub fn show_config(
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

    pub fn load_dqn_config<'a>(model_dir: impl Into<&'a str>) -> Result<DqnConfig<Cnn>> {
        let config_path = format!("{}/agent.yaml", model_dir.into());
        let file = std::fs::File::open(config_path.clone())?;
        let rdr = std::io::BufReader::new(file);
        let config: DqnAtariAgentConfig = serde_yaml::from_reader(rdr)?;
        println!("Load agent config: {}", config_path);
        Ok(config.into())
    }

    pub fn load_async_trainer_config<'a>(
        model_dir: impl Into<&'a str>,
    ) -> Result<AsyncTrainerConfig> {
        let config_path = format!("{}/trainer.yaml", model_dir.into());
        let file = std::fs::File::open(config_path.clone())?;
        let rdr = std::io::BufReader::new(file);
        let config: DqnAtariAsyncTrainerConfig = serde_yaml::from_reader(rdr)?;
        println!("Load async trainer config: {}", config_path);
        Ok(config.into())
    }

    pub fn load_replay_buffer_config<'a>(
        model_dir: impl Into<&'a str>,
    ) -> Result<SimpleReplayBufferConfig> {
        let config_path = format!("{}/replay_buffer.yaml", model_dir.into());
        let file = std::fs::File::open(config_path.clone())?;
        let rdr = std::io::BufReader::new(file);
        let config: DqnAtariReplayBufferConfig = serde_yaml::from_reader(rdr)?;
        println!("Load replay buffer config: {}", config_path);
        Ok(config.into())
    }

    pub fn create_async_trainer_config(matches: &ArgMatches) -> Result<()> {
        let model_dir = utils::model_dir(matches);
        let config = DqnAtariAsyncTrainerConfig::default();
        let path = model_dir + "/trainer.yaml";
        let mut file = std::fs::File::create(path.clone())?;
        file.write_all(serde_yaml::to_string(&config)?.as_bytes())?;
        println!("Create trainer config file: {}", path);
        Ok(())
    }

    pub fn create_replay_buffer_config(matches: &ArgMatches) -> Result<()> {
        let model_dir = utils::model_dir(matches);
        let config = util_dqn_atari::DqnAtariReplayBufferConfig::default();
        let path = model_dir + "/replay_buffer.yaml";
        let mut file = std::fs::File::create(path.clone())?;
        file.write_all(serde_yaml::to_string(&config)?.as_bytes())?;
        println!("Create replay buffer config file: {}", path);
        Ok(())
    }

    pub fn create_agent_config(matches: &ArgMatches) -> Result<()> {
        let model_dir = utils::model_dir(matches);
        let config = util_dqn_atari::DqnAtariAgentConfig::default();
        let path = model_dir + "/agent.yaml";
        let mut file = std::fs::File::create(path.clone())?;
        file.write_all(serde_yaml::to_string(&config)?.as_bytes())?;
        println!("Create agent config file: {}", path);
        Ok(())
    }

    #[derive(Serialize)]
    pub struct DqnAtariAsyncConfig {
        pub trainer: AsyncTrainerConfig,
        pub replay_buffer: SimpleReplayBufferConfig,
        pub agent: DqnConfig<Cnn>,
    }
}

mod utils {
    use super::*;

    pub fn model_dir(matches: &ArgMatches) -> String {
        let name = matches
            .value_of("name")
            .expect("The name of the environment was not given")
            .to_string();
        format!("./border/examples/atari/model/tch/dqn_{}_async", name)

        // let name = matches
        //     .value_of("name")
        //     .expect("The name of the environment was not given")
        //     .to_string();
        // let mut params = Params::default();

        // if matches.is_present("ddqn") {
        //     params = params.ddqn();
        // }

        // if matches.is_present("per") {
        //     params = params.per();
        // }

        // if matches.is_present("debug") {
        //     params = params.debug();
        // }

        // let model_dir = model_dir_async_(name, &params)?;

        // Ok(model_dir)
    }

    pub fn n_actions(env_config: &EnvConfig) -> Result<usize> {
        Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
    }

    pub fn create_recorder(
        matches: &ArgMatches,
        model_dir: &str,
        config: &DqnAtariAsyncConfig,
    ) -> Result<Box<dyn AggregateRecorder>> {
        match matches.is_present("mlflow") {
            true => {
                let name = matches.value_of("name").unwrap();
                let client = MlflowTrackingClient::new("http://localhost:8080")
                    .set_experiment_id("Atari")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", name)?;
                recorder_run.set_tag("algo", "dqn_async")?;
                recorder_run.set_tag("backend", "tch")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }

    pub fn create_matches<'a>() -> ArgMatches<'a> {
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
                Arg::with_name("create-config")
                    .long("create-config")
                    .help("Create config files"),
            )
            // .arg(
            //     Arg::with_name("per")
            //         .long("per")
            //         .takes_value(false)
            //         .help("Train/play with prioritized experience replay"),
            // )
            // .arg(
            //     Arg::with_name("ddqn")
            //         .long("ddqn")
            //         .takes_value(false)
            //         .help("Train/play with double DQN"),
            // )
            // .arg(
            //     Arg::with_name("debug")
            //         .long("debug")
            //         .takes_value(false)
            //         .help("Run with debug configuration"),
            // )
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
            .arg(
                Arg::with_name("mlflow")
                    .long("mlflow")
                    .help("Logging with mlflow"),
            )
            .get_matches();

        matches
    }
}

fn train(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = utils::model_dir(&matches);
    let env_config_train = config::env_config(name);
    let n_actions = utils::n_actions(&env_config_train)?;

    // exploration parameters
    let n_actors = matches
        .value_of("n-actors")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let eps_min = matches.value_of("eps-min").unwrap().parse::<f64>().unwrap();
    let eps_max = matches.value_of("eps-max").unwrap().parse::<f64>().unwrap();

    // Configurations
    let agent_config = config::load_dqn_config(model_dir.as_str())?
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
    let env_config_eval = config::env_config(name).eval();
    let replay_buffer_config = config::load_replay_buffer_config(model_dir.as_str())?;
    let step_proc_config = SimpleStepProcessorConfig::default();
    let actor_man_config = ActorManagerConfig::default();
    let async_trainer_config = config::load_async_trainer_config(model_dir.as_str())?;

    if matches.is_present("show-config") {
        config::show_config(
            &env_config_train,
            &agent_config,
            &actor_man_config,
            &async_trainer_config,
        );
    } else {
        let config = config::DqnAtariAsyncConfig {
            trainer: async_trainer_config.clone(),
            replay_buffer: replay_buffer_config.clone(),
            agent: agent_config.clone(),
        };
        let mut recorder = utils::create_recorder(&matches, &model_dir, &config)?;
        let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;

        train_async::<Agent, Env, ReplayBuffer, StepProc>(
            &agent_config,
            &agent_configs,
            &env_config_train,
            &env_config_eval,
            &step_proc_config,
            &replay_buffer_config,
            &actor_man_config,
            &async_trainer_config,
            &mut recorder,
            &mut evaluator,
        );
    }

    Ok(())
}

fn create_config(matches: ArgMatches) -> Result<()> {
    config::create_async_trainer_config(&matches)?;
    config::create_replay_buffer_config(&matches)?;
    config::create_agent_config(&matches)?;
    Ok(())
}

fn main() -> Result<()> {
    tch::set_num_threads(1);
    let matches = utils::create_matches();

    if matches.is_present("create-config") {
        create_config(matches)?;
    } else {
        train(matches)?;
    }

    Ok(())
}
