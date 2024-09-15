use crate::{
    Actor, ActorManagerConfig, ActorStat, PushedItemMessage, ReplayBufferProxyConfig, SyncModel,
};
use border_core::{
    Agent, Configurable, Env, ExperienceBufferBase, ReplayBufferBase, StepProcessor,
};
use crossbeam_channel::{bounded, /*unbounded,*/ Receiver, Sender};
use log::info;
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

/// Manages [`Actor`]s.
///
/// This struct handles the following requests:
/// * From the [`AsyncTrainer`] for updating the latest model info, stored in this struct.
/// * From the [`Actor`]s for getting the latest model info.
/// * From the [`Actor`]s for pushing sample batch to the `LearnerManager`.
///
/// [`AsyncTrainer`]: crate::AsyncTrainer
pub struct ActorManager<A, E, R, P>
where
    A: Agent<E, R> + Configurable + SyncModel,
    E: Env,
    P: StepProcessor<E>,
    R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
{
    /// Configurations of [`Agent`]s.
    agent_configs: Vec<A::Config>,

    /// Configuration of [`Env`].
    env_config: E::Config,

    /// Configuration of a `StepProcessor`.
    step_proc_config: P::Config,

    /// Thread handles.
    threads: Vec<JoinHandle<()>>,

    /// Number of samples to be buffered in each actor before being pushed to the replay buffer.
    ///
    /// This parameter is used as `n_buffer` in [`ReplayBufferProxyConfig`].
    n_buffer: usize,

    /// Flag to stop training
    stop: Arc<Mutex<bool>>,

    /// Receiver of [PushedItemMessage]s from [Actor].
    batch_message_receiver: Option<Receiver<PushedItemMessage<R::Item>>>,

    /// Sender of [PushedItemMessage]s to [AsyncTrainer](crate::AsyncTrainer).
    pushed_item_message_sender: Sender<PushedItemMessage<R::Item>>,

    /// Information of the model.
    ///
    /// Also has The number of optimization steps of the current model information.
    model_info: Option<Arc<Mutex<(usize, A::ModelInfo)>>>,

    /// Receives incoming model info from [AsyncTrainer](crate::AsyncTrainer).
    model_info_receiver: Receiver<(usize, A::ModelInfo)>,

    /// Stats of [Actor]s, shared with actor threads.
    actor_stats: Vec<Arc<Mutex<Option<ActorStat>>>>,

    phantom: PhantomData<R>,
}

impl<A, E, R, P> ActorManager<A, E, R, P>
where
    A: Agent<E, R> + Configurable + SyncModel,
    E: Env,
    P: StepProcessor<E>,
    R: ExperienceBufferBase<Item = P::Output> + Send + 'static + ReplayBufferBase,
    A::Config: Send + 'static,
    E::Config: Send + 'static,
    P::Config: Send + 'static,
    R::Item: Send + 'static,
    A::ModelInfo: Send + 'static,
{
    /// Builds a [`ActorManager`].
    pub fn build(
        config: &ActorManagerConfig,
        agent_configs: &Vec<A::Config>,
        env_config: &E::Config,
        step_proc_config: &P::Config,
        pushed_item_message_sender: Sender<PushedItemMessage<R::Item>>,
        model_info_receiver: Receiver<(usize, A::ModelInfo)>,
        stop: Arc<Mutex<bool>>,
    ) -> Self {
        Self {
            agent_configs: agent_configs.clone(),
            env_config: env_config.clone(),
            step_proc_config: step_proc_config.clone(),
            n_buffer: config.n_buffer,
            stop,
            threads: vec![],
            batch_message_receiver: None,
            pushed_item_message_sender,
            model_info: None,
            model_info_receiver,
            actor_stats: vec![],
            phantom: PhantomData,
        }
    }

    /// Runs threads for [`Actor`]s and a thread for sending samples into the replay buffer.
    ///
    /// Each thread is blocked until receiving the initial [`SyncModel::ModelInfo`]
    /// from [`AsyncTrainer`](crate::AsyncTrainer).
    pub fn run(&mut self, guard_init_env: Arc<Mutex<bool>>) {
        // Guard for sync of the initial model
        let guard_init_model = Arc::new(Mutex::new(true));

        // Dummy model info
        self.model_info = {
            let agent = A::build(self.agent_configs[0].clone());
            Some(Arc::new(Mutex::new(agent.model_info())))
        };

        // Thread for waiting [SyncModel::ModelInfo]
        {
            let stop = self.stop.clone();
            let model_info_receiver = self.model_info_receiver.clone();
            let model_info = self.model_info.as_ref().unwrap().clone();
            let guard_init_model = guard_init_model.clone();
            let handle = std::thread::spawn(move || {
                Self::run_model_info_loop(model_info_receiver, model_info, stop, guard_init_model);
            });
            self.threads.push(handle);
            info!("Starts thread for updating model info");
        }

        // Create channel for [BatchMessage]
        // let (s, r) = unbounded();
        let (s, r) = bounded(1000);
        self.batch_message_receiver = Some(r.clone());

        // Runs sampling processes
        self.agent_configs
            .clone()
            .into_iter()
            .enumerate()
            .for_each(|(id, agent_config)| {
                let sender = s.clone();
                let replay_buffer_proxy_config = ReplayBufferProxyConfig {
                    n_buffer: self.n_buffer,
                };
                let env_config = self.env_config.clone();
                let step_proc_config = self.step_proc_config.clone();
                let stop = self.stop.clone();
                let seed = id;
                let guard = guard_init_env.clone();
                let guard_init_model = guard_init_model.clone();
                let model_info = self.model_info.as_ref().unwrap().clone();
                let stats = Arc::new(Mutex::new(None));
                self.actor_stats.push(stats.clone());

                // Spawn actor thread
                let handle = std::thread::spawn(move || {
                    Actor::<A, E, P, R>::build(
                        id,
                        agent_config,
                        env_config,
                        step_proc_config,
                        replay_buffer_proxy_config,
                        stop,
                        seed as i64,
                        stats,
                    )
                    .run(sender, model_info, guard, guard_init_model);
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
    pub fn join(self) -> Vec<ActorStat> {
        for h in self.threads {
            h.join().unwrap();
        }

        self.actor_stats
            .iter()
            .map(|e| e.lock().unwrap().clone().unwrap())
            .collect::<Vec<_>>()
    }

    /// Stops actor threads.
    pub fn stop(&self) {
        let mut stop = self.stop.lock().unwrap();
        *stop = true;
    }

    /// Stops and joins actors.
    pub fn stop_and_join(self) -> Vec<ActorStat> {
        self.stop();
        self.join()
    }

    /// Loop waiting [PushedItemMessage] from [Actor]s.
    fn handle_message(
        receiver: Receiver<PushedItemMessage<R::Item>>,
        stop: Arc<Mutex<bool>>,
        sender: Sender<PushedItemMessage<R::Item>>,
    ) {
        let mut _n_samples = 0;

        loop {
            // Handle incoming message
            // TODO: error handling, timeout
            // TODO: caching
            // TODO: stats
            let msg = receiver.recv();
            if msg.is_ok() {
                _n_samples += 1;
                sender.try_send(msg.unwrap()).unwrap();    
            }

            // Stop the loop
            if *stop.lock().unwrap() {
                break;
            }
        }
        info!("Stopped thread for message handling");
    }

    fn run_model_info_loop(
        model_info_receiver: Receiver<(usize, A::ModelInfo)>,
        model_info: Arc<Mutex<(usize, A::ModelInfo)>>,
        stop: Arc<Mutex<bool>>,
        guard_init_model: Arc<Mutex<bool>>,
    ) {
        // Blocks threads sharing model_info until arriving the first message from AsyncTrainer.
        {
            let mut guard_init_model = guard_init_model.lock().unwrap();
            let mut model_info = model_info.lock().unwrap();
            // TODO: error handling
            let msg = model_info_receiver.recv().unwrap();
            assert_eq!(msg.0, 0);
            *model_info = msg;
            *guard_init_model = true;
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
        info!("Stopped model info thread");
    }
}
