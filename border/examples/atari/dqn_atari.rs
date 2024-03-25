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

use obs_act_types::*;

mod config {
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
        DqnConfig::<Cnn>::load(config_path)
    }

    pub fn load_trainer_config<'a>(model_dir: impl Into<&'a str>) -> Result<TrainerConfig> {
        let config_path = format!("{}/trainer.yaml", model_dir.into());
        TrainerConfig::load(config_path)
    }

    pub fn load_replay_buffer_config<'a>(
        model_dir: impl Into<&'a str>,
    ) -> Result<SimpleReplayBufferConfig> {
        let config_path = format!("{}/replay_buffer.yaml", model_dir.into());
        SimpleReplayBufferConfig::load(config_path)
    }
}

mod utils {
    use super::*;

    pub fn model_dir(matches: &ArgMatches) -> String {
        let name = matches
            .value_of("name")
            .expect("The name of the environment was not given")
            .to_string();
        format!("./border/examples/atari/model/dqn_{}", name)

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
        model_dir: &str,
        mlflow: bool,
        config: &TrainerConfig,
    ) -> Result<Box<dyn Recorder>> {
        match mlflow {
            true => {
                let client = MlflowTrackingClient::new("http://localhost:8080")
                    .set_experiment_id("Default")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
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
            .get_matches();

        matches
    }
}

fn train(matches: ArgMatches) -> Result<()> {
    // Configurations
    let mlflow = matches.is_present("mlflow");
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
        let mut agent = Dqn::build(agent_config);
        let mut recorder = utils::create_recorder(&model_dir, mlflow, &trainer_config)?;
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

fn main() -> Result<()> {
    let matches = utils::create_matches();

    if matches.is_present("play") || matches.is_present("play-gdrive") {
        play(matches)?;
    } else {
        train(matches)?;
    }

    Ok(())
}
