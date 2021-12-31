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
pub use actor::Actor;
pub use actor_manager::{ActorManager, ActorManagerConfig};
pub use async_trainer::{AsyncTrainer, AsyncTrainerConfig};
pub use messages::PushedItemMessage;
pub use replay_buffer_proxy::{ReplayBufferProxy, ReplayBufferProxyConfig};

#[cfg(test)]
mod test {
    use super::{ActorManager, ActorManagerConfig};
    use border_atari_env::util::test::*;
    use border_core::{
        replay_buffer::{SimpleStepProcessor, SimpleStepProcessorConfig},
        Env as _,
    };

    type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;

    fn actor_man_config() -> ActorManagerConfig<Agent, Env, StepProc, ReplayBuffer> {
        let n_actors = 1;
        let env_config = env_config("pong".to_string());
        let env = Env::build(&env_config, 0).unwrap();
        let n_acts = env.get_num_actions_atari() as _;
        let agent_config = RandomAgentConfig { n_acts };
        let step_proc_config = SimpleStepProcessorConfig::default();

        ActorManagerConfig::new(n_actors, agent_config, env_config, step_proc_config)
    }

    #[test]
    fn test_actor_manager() {
        let config = actor_man_config();
        let mut man = ActorManager::build(&config);
        man.run();
        std::thread::sleep(std::time::Duration::from_secs(5));
        man.stop();
        man.join();
    }
}
