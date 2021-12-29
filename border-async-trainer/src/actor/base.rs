// use std::marker::PhantomData;
// use log::info;
// use tokio::sync::broadcast;
// use std::future::{Future, Ready};
// use async_trait::async_trait;
use border_core::{Agent, Env, ReplayBufferBase};
use std::{marker::PhantomData, sync::{Arc, Mutex}};

pub struct Actor<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase,
{
    /// Stops sampling process if this field is set to `true`.
    stop: Arc<Mutex<bool>>,
    agent: A,
    env: E,
    samples_per_push: usize,
    phantom: PhantomData<(A, E, R)>,
}

impl<A, E, R> Actor<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase,
{
    pub fn build(
        agent_config: A::Config,
        env_config: E::Config,
        samples_per_push: usize,
        stop: Arc<Mutex<bool>>,
        seed: i64) -> Self {

        let agent = A::build(agent_config);
        let env = E::build(&env_config, seed).unwrap();

        Self {
            stop,
            agent,
            env,
            samples_per_push,
            phantom: PhantomData
        }
    }

    pub fn run(&mut self) {
        loop {
            if *self.stop.lock().unwrap() {
                break;
            }
        }
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
