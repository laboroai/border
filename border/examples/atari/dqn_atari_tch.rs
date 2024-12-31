mod util_dqn_atari;
use anyhow::Result;
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
    Configurable, DefaultEvaluator, Env as _, Evaluator as _, ReplayBufferBase, StepProcessor,
    Trainer, TrainerConfig,
};
use border_derive::{Act, BatchBase};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tch_agent::{
    cnn::Cnn,
    dqn::{Dqn as Dqn_, DqnConfig},
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
    pub type Dqn = Dqn_<Env, Cnn, ReplayBuffer>;
    pub type Evaluator = DefaultEvaluator<Env>;
}

use config::DqnAtariConfig;
use obs_act_types::*;

fn cuda_if_available() -> tch::Device {
    tch::Device::cuda_if_available()
}

mod config {
    use self::util_dqn_atari::{
        DqnAtariAgentConfig, DqnAtariReplayBufferConfig, DqnAtariTrainerConfig,
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
        trainer_config: &TrainerConfig,
    ) {
        println!("Device: {:?}", cuda_if_available());
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

    pub fn create_trainer_config(args: &Args) -> Result<()> {
        let model_dir = utils::model_dir(args);
        let config = util_dqn_atari::DqnAtariTrainerConfig::default();
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
    pub struct DqnAtariConfig {
        pub trainer: TrainerConfig,
        pub replay_buffer: SimpleReplayBufferConfig,
        pub agent: DqnConfig<Cnn>,
    }
}

mod utils {
    use super::*;

    pub fn model_dir(args: &Args) -> String {
        let name = &args.name;
        format!("./border/examples/atari/model/tch/dqn_{}", name)
    }

    pub fn model_dir_for_eval(args: &Args) -> String {
        model_dir(args)
    }

    pub fn n_actions(env_config: &EnvConfig) -> Result<usize> {
        Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
    }

    pub fn create_recorder(
        args: &Args,
        model_dir: &str,
        config: Option<&DqnAtariConfig>,
    ) -> Result<Box<dyn Recorder<Env, ReplayBuffer>>> {
        match args.mlflow {
            true => {
                let name = &args.name;
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment("Atari")?;
                let recorder_run = client.create_recorder(format!("{}_tch", name))?;
                if let Some(config) = config {
                    recorder_run.log_params(&config)?;
                    recorder_run.set_tag("env", name)?;
                    recorder_run.set_tag("algo", "dqn")?;
                    recorder_run.set_tag("backend", "tch")?;
                }
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(
                model_dir, model_dir, false,
            ))),
        }
    }
}

/// Train/eval DQN agent in atari environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Name of the game
    name: String,

    /// Train DQN agent, not evaluate
    #[arg(long, default_value_t = false)]
    train: bool,

    /// Evaluate DQN agent, not train
    #[arg(long, default_value_t = false)]
    eval: bool,

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
}

fn train(args: &Args) -> Result<()> {
    // Configurations
    let name = &args.name;
    let model_dir = utils::model_dir(&args);
    let env_config_train = config::env_config(name);
    let env_config_eval = config::env_config(name).eval();
    let n_actions = utils::n_actions(&env_config_train)?;
    let agent_config = {
        let agent_config = config::load_dqn_config(model_dir.as_str())?
            .out_dim(n_actions as _)
            .device(cuda_if_available());
        agent_config
    };
    let trainer_config = config::load_trainer_config(model_dir.as_str())?;
    let replay_buffer_config = config::load_replay_buffer_config(model_dir.as_str())?;
    let step_proc_config = SimpleStepProcessorConfig {};

    // Show configs or train
    if args.show_config {
        config::show_config(&env_config_train, &agent_config, &trainer_config);
    } else {
        let config = DqnAtariConfig {
            trainer: trainer_config.clone(),
            replay_buffer: replay_buffer_config.clone(),
            agent: agent_config.clone(),
        };
        let mut trainer = Trainer::build(trainer_config);
        let env = Env::build(&env_config_train, 0)?;
        let step_proc = StepProc::build(&step_proc_config);
        let mut agent = Box::new(Dqn::build(agent_config)) as _;
        let mut buffer = ReplayBuffer::build(&replay_buffer_config);
        let mut recorder = utils::create_recorder(&args, &model_dir, Some(&config))?;
        let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;

        trainer.train(
            env,
            step_proc,
            &mut agent,
            &mut buffer,
            &mut recorder,
            &mut evaluator,
        )?;
    }

    Ok(())
}

fn eval(args: &Args) -> Result<()> {
    let name = &args.name;
    let model_dir = utils::model_dir_for_eval(&args);

    let (env_config, n_actions) = {
        let env_config = config::env_config(name).render(true);
        let n_actions = utils::n_actions(&env_config)?;
        (env_config, n_actions)
    };
    let mut agent = {
        let device = cuda_if_available();
        let agent_config = config::load_dqn_config(model_dir.as_str())?
            .out_dim(n_actions as _)
            .device(device);
        let mut agent = Box::new(Dqn::build(agent_config)) as _;
        let recorder = utils::create_recorder(&args, model_dir.as_ref(), None)?;
        recorder.load_model("best".as_ref(), &mut agent)?;
        agent.eval();
        agent
    } as _;
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn create_config(args: &Args) -> Result<()> {
    config::create_trainer_config(&args)?;
    config::create_replay_buffer_config(&args)?;
    config::create_agent_config(&args)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.eval {
        eval(&args)?;
    } else if args.create_config {
        create_config(&args)?;
    } else {
        train(&args)?;
    }

    Ok(())
}
