//! Asynchronous trainer using replay buffer.
//!
//! # Messages
//! * From Learner ([Agent](border_core::Agent)) to Actor (`Sampler`)
//!   - `ModelParams`
// mod actor;
mod actor;
mod actor_manager;
mod async_trainer;
mod messages;
mod replay_buffer_proxy;
mod sync_model;
pub use actor::Actor;
pub use actor_manager::{ActorManager, ActorManagerConfig};
pub use async_trainer::{AsyncTrainer, AsyncTrainerConfig};
pub use messages::PushedItemMessage;
pub use replay_buffer_proxy::{ReplayBufferProxy, ReplayBufferProxyConfig};
pub use sync_model::SyncModel;

#[cfg(test)]
mod test {
    use super::{ActorManager, ActorManagerConfig, AsyncTrainer, AsyncTrainerConfig, SyncModel};
    use border_atari_env::util::test::*;
    use border_core::{
        record::BufferedRecorder,
        replay_buffer::{
            SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
            SimpleStepProcessorConfig,
        },
        Env as _,
    };
    use crossbeam_channel::unbounded;
    use log::info;
    use std::sync::{Arc, Mutex};
    use test_log::test;

    fn replay_buffer_config() -> SimpleReplayBufferConfig {
        SimpleReplayBufferConfig::default()
    }

    fn actor_man_config(n_actors: usize) -> ActorManagerConfig {
        ActorManagerConfig::new(n_actors)
    }

    fn async_trainer_config() -> AsyncTrainerConfig {
        AsyncTrainerConfig {
            model_dir: Some("".to_string()),
            record_interval: 10,
            eval_interval: 10,
            max_train_steps: 100,
            save_interval: 100,
            sync_interval: 10,
            eval_episodes: 1,
        }
    }

    // #[test]
    // fn test_actor_manager() {
    //     type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    //     type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    //     type ActorManager_ = ActorManager<RandomAgent, Env, ReplayBuffer, StepProc>;

    //     let env_config = env_config("pong".to_string());
    //     let env = Env::build(&env_config, 0).unwrap();
    //     let n_acts = env.get_num_actions_atari() as _;
    //     let agent_config = RandomAgentConfig { n_acts };
    //     let step_proc_config = SimpleStepProcessorConfig::default();
    //     let actor_man_config = actor_man_config(2);

    //     // Pushed items into replay buffer
    //     let (item_s, item_r) = unbounded();

    //     let mut actors = ActorManager_::build(
    //         &actor_man_config, &agent_config, &env_config, &step_proc_config
    //     );
    //     actors.run();
    //     std::thread::sleep(std::time::Duration::from_secs(5));
    //     actors.stop();
    //     actors.join();
    // }

    impl SyncModel for RandomAgent {
        type ModelInfo = ();

        fn model_info(&self) -> (usize, Self::ModelInfo) {
            info!("Returns the current model info");
            (self.n_opts_steps(), ())
        }

        fn sync_model(&mut self, _model_info: &Self::ModelInfo) {
            info!("Sync model");
        }
    }

    #[test]
    fn test_async_trainer() {
        type Agent = RandomAgent;
        type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
        type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
        type ActorManager_ = ActorManager<RandomAgent, Env, ReplayBuffer, StepProc>;
        type AsyncTrainer_ = AsyncTrainer<Agent, Env, ReplayBuffer>;

        let env_config = env_config("pong".to_string());
        let env = Env::build(&env_config, 0).unwrap();
        let n_acts = env.get_num_actions_atari() as _;
        let agent_config = RandomAgentConfig { n_acts };
        let step_proc_config = SimpleStepProcessorConfig::default();
        let replay_buffer_config = replay_buffer_config();
        let actor_man_config = actor_man_config(1);
        let async_trainer_config = async_trainer_config();

        let mut recorder = BufferedRecorder::new();

        // Pushed items into replay buffer
        let (item_s, item_r) = unbounded();

        // Synchronizing model
        let (model_s, model_r) = unbounded();

        // Prevents simlutaneous initialization of env
        let guard_init_env = Arc::new(Mutex::new(true));

        let mut actors = ActorManager_::build(
            &actor_man_config,
            &agent_config,
            &env_config,
            &step_proc_config,
            item_s,
            model_r,
        );
        let mut trainer = AsyncTrainer_::build(
            &async_trainer_config,
            &agent_config,
            &env_config,
            &replay_buffer_config,
            item_r,
            model_s,
        );

        actors.run(guard_init_env.clone());
        trainer.train(&mut recorder, guard_init_env);

        actors.stop();
        actors.join();
    }
}
