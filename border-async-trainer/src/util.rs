//! Utility function.
use crate::{
    actor_stats_fmt, ActorManager, ActorManagerConfig, AsyncTrainer, AsyncTrainerConfig, SyncModel,
};
use border_core::{
    record::TensorboardRecorder, Agent, DefaultEvaluator, Env, ReplayBufferBase,
    StepProcessorBase,
};
use crossbeam_channel::unbounded;
use log::info;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

/// Runs asynchronous training.
///
/// This function runs [`ActorManager`] and [`AsyncTrainer`] on threads.
/// These communicate using [`crossbeam_channel`]. Training logs are recorded for
/// tensorboard.
///
/// * `model_dir` - Directory where trained models and tensor board log will be saved.
/// * `agent_config` - Configuration of the agent to be trained.
/// * `agent_configs` - Configurations of agents for asynchronous sampling.
///   It must share the same structure of the model ([`SyncModel::ModelInfo`]),
///   while exploration parameters can be different.
/// * `env_config_train` - Configuration of the environment with which transitions are
///   sampled.
/// * `env_config_eval` - Configuration of the environment on which the agent being trained
///   is evaluated.
/// * `replay_buffer_config` - Configuration of the replay buffer.
/// * `actor_man_config` - Configuration of [`ActorManager`].
/// * `async_trainer_config` - Configuration of [`AsyncTrainer`].
pub fn train_async<A, E, R, S, P>(
    model_dir: &P,
    agent_config: &A::Config,
    agent_configs: &Vec<A::Config>,
    env_config_train: &E::Config,
    env_config_eval: &E::Config,
    step_proc_config: &S::Config,
    replay_buffer_config: &R::Config,
    actor_man_config: &ActorManagerConfig,
    async_trainer_config: &AsyncTrainerConfig,
) where
    A: Agent<E, R> + SyncModel,
    E: Env,
    R: ReplayBufferBase<PushedItem = S::Output> + Send + 'static,
    S: StepProcessorBase<E>,
    A::Config: Send + 'static,
    E::Config: Send + 'static,
    S::Config: Send + 'static,
    R::PushedItem: Send + 'static,
    A::ModelInfo: Send + 'static,
    P: AsRef<Path>,
{
    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut evaluator = DefaultEvaluator::new(env_config_eval, 0, 1).unwrap();

    // Shared flag to stop actor threads
    let stop = Arc::new(Mutex::new(false));

    // Creates channels
    let (item_s, item_r) = unbounded(); // items pushed to replay buffer
    let (model_s, model_r) = unbounded(); // model_info

    // guard for initialization of envs in multiple threads
    let guard_init_env = Arc::new(Mutex::new(true));

    // Actor manager and async trainer
    let mut actors = ActorManager::<A, E, R, S>::build(
        actor_man_config,
        agent_configs,
        env_config_train,
        step_proc_config,
        item_s,
        model_r,
        stop.clone(),
    );
    let mut trainer = AsyncTrainer::<A, E, R>::build(
        async_trainer_config,
        agent_config,
        env_config_eval,
        replay_buffer_config,
        item_r,
        model_s,
        stop.clone(),
    );

    // Starts sampling and training
    actors.run(guard_init_env.clone());
    let stats = trainer.train(&mut recorder, &mut evaluator, guard_init_env);
    info!("Stats of async trainer");
    info!("{}", stats.fmt());

    let stats = actors.stop_and_join();
    info!("Stats of generated samples in actors");
    info!("{}", actor_stats_fmt(&stats));
}
