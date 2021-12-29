use std::marker::PhantomData;

use border_core::{Env, Agent, ReplayBufferBase};

/// Configuration of [ActorManager](super::ActorManager).
#[derive(Clone, Debug)]
pub struct ActorManagerConfig<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase,
{
    /// The number of [Actor]s.
    pub n_actors: usize,

    /// Configuration of [Agent].
    pub agent_config: A::Config,

    /// Configuration of [Env].
    pub env_config: E::Config,

    /// Number of samples to be buffered in each actor until being pushed to the replay buffer.
    ///
    /// The default value is 100.
    pub samples_per_push: usize,

    phantom: PhantomData<R>
}

impl<A, E, R> ActorManagerConfig<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase,
{
    pub fn new(n_actors: usize, agent_config: A::Config, env_config: E::Config) -> Self {
        Self {
            n_actors,
            agent_config,
            env_config,
            samples_per_push: 100,
            phantom: PhantomData,
        }
    }
}
