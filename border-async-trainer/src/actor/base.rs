// use std::marker::PhantomData;
// use log::info;
// use tokio::sync::broadcast;
// use std::future::{Future, Ready};
// use async_trait::async_trait;
use border_core::{Agent, Env, ReplayBufferBase, SyncSampler, StepProcessorBase};
use std::{marker::PhantomData, sync::{Arc, Mutex}};
use crate::{ReplayBufferProxy, ReplayBufferProxyConfig};

/// Runs interaction between an [Agent] and an [Env], taking samples.
///
/// Samples taken will be pushed into a replay buffer via [ActorManager](crate::ActorManager).
pub struct Actor<A, E, P, R>
where
    A: Agent<E, R>,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// Stops sampling process if this field is set to `true`.
    #[allow(dead_code)] // TODO: remove this
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
    A: Agent<E, R>,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    pub fn build(
        agent_config: A::Config,
        env_config: E::Config,
        step_proc_config: P::Config,
        replay_buffer_config: ReplayBufferProxyConfig,
        samples_per_push: usize,
        stop: Arc<Mutex<bool>>,
        env_seed: i64) -> Self {

        Self {
            stop,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            step_proc_config: step_proc_config.clone(),
            replay_buffer_config: replay_buffer_config.clone(),
            samples_per_push,
            env_seed,
            phantom: PhantomData
        }
    }

    /// Runs sampling loop until `self.stop` becomes `true`.
    #[allow(unused_variables, unused_mut)] // TODO: remove this
    pub fn run(&mut self) {
        let mut agent = A::build(self.agent_config.clone());
        let mut env = E::build(&self.env_config, self.env_seed).unwrap();
        let mut step_proc = P::build(&self.step_proc_config);
        let mut buffer = ReplayBufferProxy::<R>::build(&self.replay_buffer_config);
        let mut sampler = SyncSampler::new(env, step_proc);
        let mut env_step = 0;

        // // Sampling loop
        // loop {
        //     // TODO: error handling
        //     let _record = sampler.sample_and_push(&mut agent, &mut buffer).unwrap();

        //     // Stop the sampling loop
        //     if *self.stop.lock().unwrap() {
        //         break;
        //     }
        // }
    }
}

// #[derive(Debug, Clone)]
// /// Commands [AsyncActor] accepts.
// pub enum AsyncActorCommand<M> {
//     /// Nothing will do.
//     None,

//     /// Runs asynchronous sampling.
//     Run,

//     /// Synchronize model.
//     Sync(M),
// }

// #[async_trait]
// /// Asynchronous experience sampling with [Agent](border_core::Agent).
// pub trait AsyncActor<M: std::fmt::Debug + Clone + Send> {
//     type ModelParams;

//     /// Waits call by the channel. This method handles [AsyncActorCommand].
//     async fn new(receiver: broadcast::Receiver<AsyncActorCommand<M>>)
//     where
//         M: 'async_trait;
// }

// /// Performes asynchronous experience sampling with agents in `border-tch-agent`.
// pub struct TchAsyncActor<M> {
//     phantom: PhantomData<M>
// }

// impl<M: std::fmt::Debug + Clone> TchAsyncActor<M> {
//     async fn handle_command(mut rx: broadcast::Receiver<AsyncActorCommand<M>>) {
//         println!("Start event loop");
//         loop {
//             let msg = rx.recv().await;
//             println!("{:?}", msg);
//         }
//     }
// }

// #[async_trait]
// impl<M: std::fmt::Debug + Clone + Send> AsyncActor<M> for TchAsyncActor<M> {
//     type ModelParams = M;

//     async fn new(rx: broadcast::Receiver<AsyncActorCommand<M>>)
//     where
//         M: 'async_trait
//     {
//         println!("Launch AsyncActor");
//         Self::handle_command(rx).await;
//     }
// }

// #[tokio::test]
// async fn test_tch_async_actor() {
//     let (tx, rx1) = broadcast::channel(16);

//     std::thread::spawn(move || {
//         for i in 0..20 {
//             std::thread::sleep(std::time::Duration::from_secs(1));
//             let _ = tx.send(AsyncActorCommand::None);
//         }
//     });

//     let handle = tokio::spawn(async move {
//         let actor = TchAsyncActor::<u8>::new(rx1).await;
//     }).await;
// }

// #[tokio::test]
// async fn test2() {
//     let (tx, mut rx) = broadcast::channel(16);

//     std::thread::spawn(move || {
//         for i in 0..20 {
//             std::thread::sleep(std::time::Duration::from_secs(1));
//             let _ = tx.send("abc");
//         }
//     });

//     let _ = tokio::spawn(async move {
//         loop {
//             let msg = rx.recv().await;
//             println!("{:?}", msg)
//         }
//     }).await;
// }
