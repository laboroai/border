// use std::marker::PhantomData;
// use log::info;
// use tokio::sync::broadcast;
// use std::future::{Future, Ready};
// use async_trait::async_trait;
use crate::{PushedItemMessage, ReplayBufferProxy, ReplayBufferProxyConfig, SyncModel};
use border_core::{Agent, Env, ReplayBufferBase, StepProcessorBase, SyncSampler};
use crossbeam_channel::Sender;
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

/// Runs interaction between an [Agent] and an [Env], taking samples.
///
/// Samples taken will be pushed into a replay buffer via [ActorManager](crate::ActorManager).
pub struct Actor<A, E, P, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// Stops sampling process if this field is set to `true`.
    id: usize,
    // #[allow(dead_code)] // TODO: remove this
    stop: Arc<Mutex<bool>>,
    agent_config: A::Config,
    env_config: E::Config,
    step_proc_config: P::Config,
    replay_buffer_config: ReplayBufferProxyConfig,
    #[allow(dead_code)] // TODO: remove this
    samples_per_push: usize,
    env_seed: i64,
    phantom: PhantomData<(A, E, P, R)>,
}

impl<A, E, P, R> Actor<A, E, P, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    pub fn build(
        id: usize,
        agent_config: A::Config,
        env_config: E::Config,
        step_proc_config: P::Config,
        replay_buffer_config: ReplayBufferProxyConfig,
        samples_per_push: usize,
        stop: Arc<Mutex<bool>>,
        env_seed: i64,
    ) -> Self {
        Self {
            id,
            stop,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            step_proc_config: step_proc_config.clone(),
            replay_buffer_config: replay_buffer_config.clone(),
            samples_per_push,
            env_seed,
            phantom: PhantomData,
        }
    }

    fn sync_model_first(agent: &mut A, model_info: &Arc<Mutex<(usize, A::ModelInfo)>>) {
        let model_info = model_info.lock().unwrap();
        agent.sync_model(&model_info.1);
    }

    fn sync_model(
        agent: &mut A,
        n_opt_steps: &mut usize,
        model_info: &Arc<Mutex<(usize, A::ModelInfo)>>,
    ) {
        let model_info = model_info.lock().unwrap();
        if model_info.0 > *n_opt_steps {
            *n_opt_steps = model_info.0;
            agent.sync_model(&model_info.1);
        }
    }

    /// Runs sampling loop until `self.stop` becomes `true`.
    #[allow(unused_variables, unused_mut)] // TODO: remove this
    pub fn run(
        &mut self,
        sender: Sender<PushedItemMessage<R::PushedItem>>,
        guard: Arc<Mutex<bool>>,
        model_info: Arc<Mutex<(usize, A::ModelInfo)>>,
    ) {
        let mut agent = A::build(self.agent_config.clone());
        let mut buffer =
            ReplayBufferProxy::<R>::build_with_sender(self.id, &self.replay_buffer_config, sender);
        let mut sampler = {
            let tmp = guard.lock().unwrap();
            let mut env = E::build(&self.env_config, self.env_seed).unwrap();
            let mut step_proc = P::build(&self.step_proc_config);
            SyncSampler::new(env, step_proc)
        };
        let mut env_step = 0;
        let mut n_opt_steps = 0;

        // Synchronize model
        Self::sync_model_first(&mut agent, &model_info);

        // Sampling loop
        loop {
            // Check model update and synchronize
            Self::sync_model(&mut agent, &mut n_opt_steps, &model_info);

            // TODO: error handling
            let _record = sampler.sample_and_push(&mut agent, &mut buffer).unwrap();

            // Stop the sampling loop
            if *self.stop.lock().unwrap() {
                break;
            }
        }
    }
}
