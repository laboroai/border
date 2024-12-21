//! aaa
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
    generic_replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::Recorder,
    DefaultEvaluator, Env as _,
};
use border_derive::{Act, BatchBase};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tch_agent::{
    cnn::Cnn,
    dqn::{Dqn, DqnConfig, DqnExplorer, EpsilonGreedy},
    TensorBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::Parser;

mod obs_act_types {
    use super::*;

    pub type Obs = BorderAtariObs;

    #[derive(Clone, BatchBase)]
    pub struct ObsBatch(TensorBatch);

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorBatch::from_tensor(tensor))
        }
    }

    #[derive(BatchBase)]
    pub struct ActBatch(TensorBatch);

    impl From<Act> for ActBatch {
        fn from(act: Act) -> Self {
            let tensor = act.into();
            Self(TensorBatch::from_tensor(tensor))
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
    pub type Evaluator = DefaultEvaluator<Env>;
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
        replay_buffer_config: &SimpleReplayBufferConfig,
        agent_config: &DqnConfig<Cnn>,
        actor_man_config: &ActorManagerConfig,
        trainer_config: &AsyncTrainerConfig,
    ) {
        println!("Device: {:?}", tch::Device::cuda_if_available());
        println!("### env_config");
        println!("{}", serde_yaml::to_string(&env_config).unwrap());
        println!("### replay_buffer_config");
        println!("{}", serde_yaml::to_string(&replay_buffer_config).unwrap());
        println!("### agent_config");
        println!("{}", serde_yaml::to_string(&agent_config).unwrap());
        println!("### actor_man_config");
        println!("{}", serde_yaml::to_string(&actor_man_config).unwrap());
        println!("### trainer_config");
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

    pub fn create_async_trainer_config(args: &Args) -> Result<()> {
        let model_dir = utils::model_dir(args);
        let config = DqnAtariAsyncTrainerConfig::default();
        let path = model_dir + "/trainer.yaml";
        let mut file = std::fs::File::create(path.clone())?;
        file.write_all(serde_yaml::to_string(&config)?.as_bytes())?;
        println!("Create trainer config file: {}", path);
        Ok(())
    }

    pub fn create_replay_buffer_config(args: &Args) -> Result<()> {
        let model_dir = utils::model_dir(args);
        let config = util_dqn_atari::DqnAtariReplayBufferConfig::default();
        let path = model_dir + "/replay_buffer.yaml";
        let mut file = std::fs::File::create(path.clone())?;
        file.write_all(serde_yaml::to_string(&config)?.as_bytes())?;
        println!("Create replay buffer config file: {}", path);
        Ok(())
    }

    pub fn create_agent_config(args: &Args) -> Result<()> {
        let model_dir = utils::model_dir(args);
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

    pub fn model_dir(args: &Args) -> String {
        let name = &args.name;
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
        args: &Args,
        model_dir: &str,
        config: &DqnAtariAsyncConfig,
    ) -> Result<Box<dyn Recorder>> {
        match args.mlflow {
            true => {
                let name = &args.name;
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment("Atari")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", name)?;
                recorder_run.set_tag("algo", "dqn_async")?;
                recorder_run.set_tag("backend", "tch")?;
                recorder_run.set_tag("n_actors", args.n_actors.to_string())?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }
}

/// Train DQN agent in atari environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Name of the game
    name: String,

    /// Create config files
    #[arg(long, default_value_t = false)]
    create_config: bool,

    /// Show config
    #[arg(long, default_value_t = false)]
    show_config: bool,

    /// Log metrics with MLflow
    #[arg(long, default_value_t = false)]
    mlflow: bool,

    /// Waiting time in milliseconds between frames when evaluation
    #[arg(long, default_value_t = 25)]
    wait: u64,

    /// Number of actors, default to 6
    #[arg(long, default_value_t = 6)]
    n_actors: usize,

    /// The minimum value of exploration noise probability, default to 0.001
    #[arg(long, default_value_t = 0.001)]
    eps_min: f64,

    /// The maximum value of exploration noise probability, default to 0.4
    #[arg(long, default_value_t = 0.4)]
    eps_max: f64,
}

fn train(args: &Args) -> Result<()> {
    let name = &args.name;
    let model_dir = utils::model_dir(&args);
    let env_config_train = config::env_config(name);
    let n_actions = utils::n_actions(&env_config_train)?;

    // exploration parameters
    let n_actors = args.n_actors;
    let eps_min = &args.eps_min;
    let eps_max = &args.eps_max;

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
    let async_trainer_config =
        config::load_async_trainer_config(model_dir.as_str())?.model_dir(model_dir.as_str())?;

    if args.show_config {
        config::show_config(
            &env_config_train,
            &replay_buffer_config,
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
        let mut recorder = utils::create_recorder(&args, &model_dir, &config)?;
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

fn create_config(args: &Args) -> Result<()> {
    config::create_async_trainer_config(&args)?;
    config::create_replay_buffer_config(&args)?;
    config::create_agent_config(&args)?;
    Ok(())
}

fn main() -> Result<()> {
    tch::set_num_threads(1);
    let args = Args::parse();

    if args.create_config {
        create_config(&args)?;
    } else {
        train(&args)?;
    }

    Ok(())
}
