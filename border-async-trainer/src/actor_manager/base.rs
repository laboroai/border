use crate::{Actor, ActorManagerConfig};
use border_core::{Agent, Env, ReplayBufferBase};
use std::{marker::PhantomData, sync::{Arc, Mutex}};

/// Manages [Actor]s.
///
/// This struct handles the following requests:
/// * From the [LearnerManager]() for updating the latest model info, stored in this struct.
/// * From the [Actor]s for getting the latest model info.
/// * From the [Actor]s for pushing sample batch to the [LearnerManager].
pub struct ActorManager<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase,
{
    /// The number of [Actor]s.
    n_actors: usize,

    /// Configuration of [Agent].
    agent_config: A::Config,

    /// Configuration of [Env].
    env_config: E::Config,

    /// Number of samples to be buffered in each actor before being pushed to the replay buffer.
    ///
    /// At the same time, [Actor] asks for [ActorManager] to get the model parameters.
    samples_per_push: usize,

    /// Flag to stop training
    stop: Arc<Mutex<bool>>,

    phantom: PhantomData<R>
}

impl<A, E, R> ActorManager<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase,
{
    /// Builds a [ActorManager].
    pub fn build(config: &ActorManagerConfig<A, E, R>) -> Self {
        Self {
            n_actors: config.n_actors,
            agent_config: config.agent_config.clone(),
            env_config: config.env_config.clone(),
            samples_per_push: config.samples_per_push,
            stop: Arc::new(Mutex::new(false)),
            phantom: PhantomData,
        }
    }

    /// Runs [Actor]s.
    pub fn run(&self) {
        // Runs sampling processes
        (0..self.n_actors).for_each(|seed| {
            Actor::<A, E, R>::build(
                self.agent_config.clone(),
                self.env_config.clone(),
                self.samples_per_push,
                self.stop.clone(),
                seed as i64,
            ).run();
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ActorManager;
    use anyhow::Result;
    use border_atari_env::BorderAtariEnvConfig;
    use border_core::{record::Record, Act, Env, Info, Obs, Step};
    // use border_core::{Env, Obs, Act, Info, Step, record::Record, Policy, Agent};

    #[derive(Clone, Debug)]
    struct TestObs {
        obs: f64,
    }

    impl Obs for TestObs {
        fn dummy(_n: usize) -> Self {
            unimplemented!();
        }

        fn merge(self, _obs_reset: Self, _is_done: &[i8]) -> Self {
            unimplemented!();
        }

        fn len(&self) -> usize {
            1
        }
    }

    #[derive(Clone, Debug)]
    struct TestAct {
        act: f64,
    }

    impl Act for TestAct {
        fn len(&self) -> usize {
            1
        }
    }

    struct TestEnv {
        #[allow(dead_code)]
        state: f64,
    }

    struct TestInfo;

    impl Info for TestInfo {}

    impl Env for TestEnv {
        type Config = ();
        type Obs = TestObs;
        type Act = TestAct;
        type Info = TestInfo;

        fn build(_config: &Self::Config, _seed: i64) -> Result<Self>
        where
            Self: Sized,
        {
            Ok(Self { state: 0f64 })
        }

        fn step_with_reset(&mut self, _a: &Self::Act) -> (Step<Self>, Record)
        where
            Self: Sized,
        {
            unimplemented!();
        }

        fn step(&mut self, _a: &Self::Act) -> (Step<Self>, Record)
        where
            Self: Sized,
        {
            unimplemented!();
        }

        fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
            unimplemented!();
        }
    }

    #[test]
    fn test_actor_manager() {
        // let actor_manager_config = ActorM
        println!("hello!");
    }
}
