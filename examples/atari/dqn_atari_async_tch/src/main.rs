mod args;
mod config;
mod types;
use anyhow::Result;
use args::Args;
use border_async_trainer::{util::train_async, ActorManagerConfig};
use border_core::{
    generic_replay_buffer::SimpleStepProcessorConfig, record::Recorder, Agent, Configurable,
    Env as _, Evaluator as _,
};
use border_mlflow_tracking::MlflowTrackingClient;
use border_tensorboard::TensorboardRecorder;
use clap::Parser;

use config::DqnAtariAsyncConfig;
use types::*;

fn create_agent(config: &DqnAtariAsyncConfig) -> Result<Box<dyn Agent<Env, ReplayBuffer>>> {
    let n_actions = Env::build(&config.env_config, 0)?.get_num_actions_atari() as i64;
    let agent_config = config.agent_config.clone().out_dim(n_actions);
    Ok(Box::new(Dqn::build(agent_config)))
}

fn create_recorder(
    args: &Args,
    model_dir: &str,
    config: Option<&DqnAtariAsyncConfig>,
) -> Result<Box<dyn Recorder<Env, ReplayBuffer>>> {
    match true {
        //args.mlflow {
        true => {
            let name = &args.name;
            let client =
                MlflowTrackingClient::new("http://localhost:8080").set_experiment("Atari")?;
            let recorder_run = client.create_recorder(format!("{}_tch", name))?;
            if let Some(config) = config {
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", name)?;
                recorder_run.set_tag("algo", "dqn_async")?;
                recorder_run.set_tag("backend", "tch")?; 
                recorder_run.set_tag("n_actors", args.n_actors.to_string())?;
            }
            Ok(Box::new(recorder_run))
        }
        false => Ok(Box::new(TensorboardRecorder::new(
            model_dir, model_dir, false,
        ))),
    }
}

pub fn n_actions(env_config: &EnvConfig) -> Result<usize> {
    Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
}

fn train(config: &DqnAtariAsyncConfig) -> Result<()> {
    let env_config_train = config.clone_env_config();
    let env_config_eval = config.clone_env_config().eval();
    let step_proc_config = SimpleStepProcessorConfig {};
    let n_actions = n_actions(&env_config_train)?;

    let mut agent_config = config.agent_config.clone();
    agent_config.model_config.q_config.as_mut().unwrap().out_dim = n_actions as i64;
    let agent_configs = (0..config.args.n_actors)
        .map(|_| {
            let mut config = agent_config.clone();
            config.device = Some(border_tch_agent::Device::Cpu);
            config
        })
        .collect::<Vec<_>>();
    let replay_buffer_config = &config.clone_replay_buffer_config();
    let actor_man_config = ActorManagerConfig::default();
    let trainer_config = config.clone_trainer_config();

    let mut recorder = create_recorder(&config.args, config.model_dir().as_ref(), Some(config))?;
    let mut evaluator = Evaluator::new(&env_config_eval, 0, 1)?;

    train_async::<Dqn, Env, types::ReplayBuffer, types::StepProc>(
        &agent_config,
        &agent_configs,
        &env_config_train,
        &env_config_eval,
        &step_proc_config,
        &replay_buffer_config,
        &actor_man_config,
        &trainer_config,
        &mut recorder,
        &mut evaluator,
    );

    Ok(())
}

fn eval(config: &DqnAtariAsyncConfig) -> Result<()> {
    let env_config = config.clone_env_config();
    let mut agent = create_agent(config)?;
    let mut evaluator = Evaluator::new(&env_config, 0, 5)?;

    // recorder is used to load model parameters
    let recorder = create_recorder(&config.args, config.model_dir().as_ref(), None)?;
    recorder.load_model("best".as_ref(), &mut agent)?;

    let _ = evaluator.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let config: DqnAtariAsyncConfig = Args::parse().into();

    match config.args.mode.as_str() {
        "train" => train(&config)?,
        "eval" => eval(&config)?,
        _ => panic!("mode must be either 'train' or 'eval'"),
    }

    Ok(())
}
