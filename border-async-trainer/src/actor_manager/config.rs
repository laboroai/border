use std::marker::PhantomData;

use border_core::{Agent, Env, ReplayBufferBase, StepProcessorBase};

/// Configuration of [ActorManager](super::ActorManager).
#[derive(Clone, Debug)]
pub struct ActorManagerConfig<A, E, P, R>
where
    A: Agent<E, R>,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// The number of [Actor]s.
    pub n_actors: usize,

    /// Configuration of [Agent].
    pub agent_config: A::Config,

    /// Configuration of [Env].
    pub env_config: E::Config,

    /// Configuration of a `StepProcessor`.
    pub step_proc_config: P::Config,

    /// Number of samples to be buffered in each actor until being pushed to the replay buffer.
    ///
    /// The default value is 100.
    pub samples_per_push: usize,

    phantom: PhantomData<R>,
}

impl<A, E, P, R> ActorManagerConfig<A, E, P, R>
where
    A: Agent<E, R>,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    pub fn new(
        n_actors: usize,
        agent_config: A::Config,
        env_config: E::Config,
        step_proc_config: P::Config,
    ) -> Self {
        Self {
            n_actors,
            agent_config,
            env_config,
            step_proc_config,
            samples_per_push: 100,
            phantom: PhantomData,
        }
    }
}
