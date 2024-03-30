mod util_dqn_atari;
use anyhow::Result;
use border_atari_env::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_candle_agent::{
    cnn::Cnn,
    dqn::{Dqn as Dqn_, DqnConfig},
    TensorSubBatch,
};
use border_core::{
    record::Recorder,
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    Agent, DefaultEvaluator, Env as _, Evaluator as _, Policy, Trainer, TrainerConfig,
};
use border_derive::{Act, SubBatch};
use border_mlflow_tracking::MlflowTrackingClient;
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
    // struct ActBatch(TensorSubBatch<ActShape, i64>);
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
    pub type Dqn = Dqn_<Env, Cnn, ReplayBuffer>;
    pub type Evaluator = DefaultEvaluator<Env, Dqn>;
}

use config::DqnAtariConfig;
use obs_act_types::*;

mod config {
    use serde::Serialize;

    use self::util_dqn_atari::{
        DqnAtariAgentConfig, DqnAtariReplayBufferConfig, DqnAtariTrainerConfig,
    };
    use std::io::Write;

    use super::*;

    pub fn env_config(name: impl Into<String>) -> EnvConfig {
        BorderAtariEnvConfig::default().name(name.into())
    }

    pub fn show_config(
        env_config: &EnvConfig,
        agent_config: &DqnConfig<Cnn>,
        trainer_config: &TrainerConfig,
    ) {
        println!("Device: {:?}", candle_core::Device::cuda_if_available(0));
        println!("{}", serde_yaml::to_string(&env_config).unwrap());
        println!("{}", serde_yaml::to_string(&agent_config).unwrap());
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

    pub fn load_trainer_config<'a>(model_dir: impl Into<&'a str>) -> Result<TrainerConfig> {
        let config_path = format!("{}/trainer.yaml", model_dir.into());
        let file = std::fs::File::open(config_path.clone())?;
        let rdr = std::io::BufReader::new(file);
        let config: DqnAtariTrainerConfig = serde_yaml::from_reader(rdr)?;
        println!("Load trainer config: {}", config_path);
        Ok(config.into())
        // TrainerConfig::load(config_path)
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

    pub fn create_trainer_config(matches: &ArgMatches) -> Result<()> {
        let model_dir = utils::model_dir(matches);
        let config = util_dqn_atari::DqnAtariTrainerConfig::default();
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
    pub struct DqnAtariConfig {
        pub trainer: TrainerConfig,
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
        format!("./border/examples/atari/model/candle/dqn_{}", name)

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

        // model_dir_(name, &params)
    }

    pub fn model_dir_for_play(matches: &ArgMatches) -> String {
        matches.value_of("play").unwrap().to_string()
    }

    pub fn n_actions(env_config: &EnvConfig) -> Result<usize> {
        Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
    }

    pub fn create_recorder(
        matches: &ArgMatches,
        model_dir: &str,
        config: &DqnAtariConfig,
    ) -> Result<Box<dyn Recorder>> {
        match matches.is_present("mlflow") {
            true => {
                let name = matches.value_of("name").unwrap();
                let client = MlflowTrackingClient::new("http://localhost:8080")
                    .set_experiment_id("Atari")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", name)?;
                recorder_run.set_tag("algo", "dqn")?;
                recorder_run.set_tag("backend", "candle")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }

    pub fn create_matches<'a>() -> ArgMatches<'a> {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        let matches = App::new("dqn_atari")
            .version("0.1.0")
            .author("Taku Yoshioka <yoshioka@laboro.ai>")
            .arg(
                Arg::with_name("name")
                    .long("name")
                    .takes_value(true)
                    .required(true)
                    .index(1)
                    .help("The name of the atari rom (e.g., pong)"),
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
                Arg::with_name("create-config")
                    .long("create-config")
                    .help("Create config files"),
            )
            // not supported yet
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
    // Configurations
    let name = matches.value_of("name").unwrap();
    let model_dir = utils::model_dir(&matches);
    let env_config_train = config::env_config(name);
    let env_config_eval = config::env_config(name).eval();
    let n_actions = utils::n_actions(&env_config_train)?;
    let agent_config = {
        let agent_config = config::load_dqn_config(model_dir.as_str())?
            .out_dim(n_actions as _)
            .device(candle_core::Device::cuda_if_available(0)?);
        agent_config
    };
    let trainer_config = config::load_trainer_config(model_dir.as_str())?;
    let replay_buffer_config = config::load_replay_buffer_config(model_dir.as_str())?;
    let step_proc_config = SimpleStepProcessorConfig {};

    // Show configs or train
    if matches.is_present("show-config") {
        config::show_config(&env_config_train, &agent_config, &trainer_config);
    } else {
        let config = DqnAtariConfig {
            trainer: trainer_config.clone(),
            replay_buffer: replay_buffer_config.clone(),
            agent: agent_config.clone(),
        };
        let mut agent = Dqn::build(agent_config);
        let mut recorder = utils::create_recorder(&matches, &model_dir, &config)?;
        let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;
        let mut trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            trainer_config,
            env_config_train,
            step_proc_config,
            replay_buffer_config,
        );

        trainer.train(&mut agent, &mut recorder, &mut evaluator)?;
    }

    Ok(())
}

fn play(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = utils::model_dir_for_play(&matches);

    let (env_config, n_actions) = {
        let env_config = config::env_config(name).render(true);
        let n_actions = utils::n_actions(&env_config)?;
        (env_config, n_actions)
    };
    let mut agent = {
        let device = candle_core::Device::cuda_if_available(0)?;
        let agent_config = config::load_dqn_config(model_dir.as_str())?
            .out_dim(n_actions as _)
            .device(device);
        let mut agent = Dqn::build(agent_config);
        agent.load(model_dir + "/best")?;
        agent.eval();
        agent
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn create_config(matches: ArgMatches) -> Result<()> {
    config::create_trainer_config(&matches)?;
    config::create_replay_buffer_config(&matches)?;
    config::create_agent_config(&matches)?;
    Ok(())
}

fn main() -> Result<()> {
    let matches = utils::create_matches();

    if matches.is_present("play") || matches.is_present("play-gdrive") {
        play(matches)?;
    } else if matches.is_present("create-config") {
        create_config(matches)?;
    } else {
        train(matches)?;
    }

    Ok(())
}
