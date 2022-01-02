use crate::{Actor, ActorManagerConfig, PushedItemMessage, ReplayBufferProxyConfig, SyncModel};
use border_core::{Agent, Env, ReplayBufferBase, StepProcessorBase};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

/// Manages [Actor]s.
///
/// This struct handles the following requests:
/// * From the [LearnerManager]() for updating the latest model info, stored in this struct.
/// * From the [Actor]s for getting the latest model info.
/// * From the [Actor]s for pushing sample batch to the `LearnerManager`.
pub struct ActorManager<A, E, R, P>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// The number of [Actor]s.
    n_actors: usize,

    /// Configuration of [Agent].
    agent_config: A::Config,

    /// Configuration of [Env].
    env_config: E::Config,

    /// Configuration of a `StepProcessor`.
    step_proc_config: P::Config,

    /// Thread handles.
    threads: Vec<JoinHandle<()>>,

    /// Number of samples to be buffered in each actor before being pushed to the replay buffer.
    ///
    /// At the same time, [Actor] asks for [ActorManager] to get the model parameters.
    samples_per_push: usize,

    /// Flag to stop training
    stop: Arc<Mutex<bool>>,

    /// Receiver of [PushedItemMessage]s from [Actor].
    batch_message_receiver: Option<Receiver<PushedItemMessage<R::PushedItem>>>,

    /// Sender of [PushedItemMessage]s to [AsyncTrainer](crate::AsyncTrainer).
    pushed_item_message_sender: Sender<PushedItemMessage<R::PushedItem>>,

    /// Information of the model.
    ///
    /// Also has The number of optimization steps of the current model information.
    model_info: Option<Arc<Mutex<(usize, A::ModelInfo)>>>,

    /// Receives incoming model info from [AsyncTrainer](crate::AsyncTrainer).
    model_info_receiver: Receiver<(usize, A::ModelInfo)>,

    phantom: PhantomData<R>,
}

impl<A, E, R, P> ActorManager<A, E, R, P>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output> + Send + 'static,
    A::Config: Send + 'static,
    E::Config: Send + 'static,
    P::Config: Send + 'static,
    R::PushedItem: Send + 'static,
    A::ModelInfo: Send + 'static,
{
    /// Builds a [ActorManager].
    pub fn build(
        config: &ActorManagerConfig,
        agent_config: &A::Config,
        env_config: &E::Config,
        step_proc_config: &P::Config,
        pushed_item_message_sender: Sender<PushedItemMessage<R::PushedItem>>,
        model_info_receiver: Receiver<(usize, A::ModelInfo)>,
    ) -> Self {
        Self {
            n_actors: config.n_actors,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            step_proc_config: step_proc_config.clone(),
            samples_per_push: config.samples_per_push,
            stop: Arc::new(Mutex::new(false)),
            threads: vec![],
            batch_message_receiver: None,
            pushed_item_message_sender,
            model_info: None,
            model_info_receiver,
            phantom: PhantomData,
        }
    }

    /// Runs threads for [Actor]s and a thread for sending samples into the replay buffer.
    ///
    /// A thread will wait for [SyncModel::ModelInfo] from [AsyncTrainer](crate::AsyncTrainer),
    /// which blocks execution of [Actor] threads.
    pub fn run(&mut self) {
        // Dummy model info
        self.model_info = {
            let agent = A::build(self.agent_config.clone());
            Some(Arc::new(Mutex::new(agent.model_info())))
        };

        // Thread for waiting [SyncModel::ModelInfo]
        {
            let stop = self.stop.clone();
            let model_info_receiver = self.model_info_receiver.clone();
            let model_info = self.model_info.as_ref().unwrap().clone();
            let handle = std::thread::spawn(move || {
                Self::run_model_info_loop(model_info_receiver, model_info, stop);
            });
            self.threads.push(handle);
        }

        // Create channel for [BatchMessage]
        let (s, r) = unbounded();
        let guard = Arc::new(Mutex::new(true));
        self.batch_message_receiver = Some(r.clone());

        // Runs sampling processes
        (0..self.n_actors).for_each(|id| {
            let sender = s.clone();
            let replay_buffer_proxy_config = ReplayBufferProxyConfig {};
            let agent_config = self.agent_config.clone();
            let env_config = self.env_config.clone();
            let step_proc_config = self.step_proc_config.clone();
            let samples_per_push = self.samples_per_push;
            let stop = self.stop.clone();
            let seed = id;
            let guard = guard.clone();
            let model_info = self.model_info.as_ref().unwrap().clone();

            // Spawn actor thread
            let handle = std::thread::spawn(move || {
                Actor::<A, E, P, R>::build(
                    id,
                    agent_config,
                    env_config,
                    step_proc_config,
                    replay_buffer_proxy_config,
                    samples_per_push,
                    stop,
                    seed as i64,
                )
                .run(sender, guard, model_info);
            });
            self.threads.push(handle);
        });

        // Thread for handling incoming samples
        {
            let stop = self.stop.clone();
            let s = self.pushed_item_message_sender.clone();
            let handle = std::thread::spawn(move || {
                Self::handle_message(r, stop, s);
            });
            self.threads.push(handle);
        }
    }

    /// Waits until all actors finish.
    pub fn join(self) {
        for h in self.threads {
            h.join().unwrap();
        }
    }

    /// Stops actor threads.
    pub fn stop(&self) {
        let mut stop = self.stop.lock().unwrap();
        *stop = true;
    }

    /// Loop waiting [PushedItemMessage] from [Actor]s.
    fn handle_message(
        receiver: Receiver<PushedItemMessage<R::PushedItem>>,
        stop: Arc<Mutex<bool>>,
        sender: Sender<PushedItemMessage<R::PushedItem>>,
    ) {
        let mut _n_samples = 0;

        loop {
            // Handle incoming message
            // TODO: error handling, timeout
            // TODO: caching
            // TODO: stats
            let msg = receiver.recv().unwrap();
            _n_samples += 1;
            sender.send(msg).unwrap();
            // println!("{:?}", (_msg.id, n_samples));

            // Stop the loop
            if *stop.lock().unwrap() {
                break;
            }
        }
    }

    fn run_model_info_loop(
        model_info_receiver: Receiver<(usize, A::ModelInfo)>,
        model_info: Arc<Mutex<(usize, A::ModelInfo)>>,
        stop: Arc<Mutex<bool>>,
    ) {
        // Blocks threads sharing model_info until arriving the first message from AsyncTrainer.
        {
            let mut model_info = model_info.lock().unwrap();
            // TODO: error handling
            let msg = model_info_receiver.recv().unwrap();
            assert_eq!(msg.0, 0);
            *model_info = msg;
        }
    
        loop {
            // TODO: error handling
            let msg = model_info_receiver.recv().unwrap();
            let mut model_info = model_info.lock().unwrap();
            *model_info = msg;
            if *stop.lock().unwrap() {
                break;
            }
        }
    }    
}
