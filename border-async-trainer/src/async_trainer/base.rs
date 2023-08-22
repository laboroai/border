use crate::{AsyncTrainerConfig, PushedItemMessage, SyncModel, AsyncTrainStat};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue::Scalar, Recorder},
    Agent, Env, Obs, ReplayBufferBase,
};
use crossbeam_channel::{Receiver, Sender};
use log::info;
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::SystemTime,
    thread::JoinHandle,
};

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Manages asynchronous training loop in a single machine.
///
/// It interacts with [`ActorManager`] as shown below:
///
/// ```mermaid
/// flowchart LR
///   subgraph ActorManager
///     E[Actor]-->|ReplayBufferBase::PushedItem|H[ReplayBufferProxy]
///     F[Actor]-->H
///     G[Actor]-->H
///   end
///   K-->|SyncModel::ModelInfo|E
///   K-->|SyncModel::ModelInfo|F
///   K-->|SyncModel::ModelInfo|G
///
///   subgraph I[AsyncTrainer]
///     H-->|PushedItemMessage|J[ReplayBuffer]
///     J-->|ReplayBufferBase::Batch|K[Agent]
///   end
/// ```
///
/// * In [`ActorManager`] (right), [`Actor`]s sample transitions, which have type
///   [`ReplayBufferBase::PushedItem`], in parallel and push the transitions into
///   [`ReplayBufferProxy`]. It should be noted that [`ReplayBufferProxy`] has a
///   type parameter of [`ReplayBufferBase`] and the proxy accepts
///   [`ReplayBufferBase::PushedItem`].
/// * The proxy sends the transitions into the replay buffer, implementing
///   [`ReplayBufferBase`], in the [`AsyncTrainer`].
/// * The [`Agent`] in [`AsyncTrainer`] trains its model parameters by using batches
///   of type [`ReplayBufferBase::Batch`], which are taken from the replay buffer.
/// * The model parameters of the [`Agent`] in [`AsyncTrainer`] are wrapped in
///   [`SyncModel::ModelInfo`] and periodically sent to the [`Agent`]s in [`Actor`]s.
///   [`Agent`] must implement [`SyncModel`] to synchronize its model.
/// 
/// [`ActorManager`]: crate::ActorManager
/// [`Actor`]: crate::Actor
/// [`ReplayBufferBase::PushedItem`]: border_core::ReplayBufferBase::PushedItem
/// [`ReplayBufferProxy`]: crate::ReplayBufferProxy
/// [`ReplayBufferBase`]: border_core::ReplayBufferBase
/// [`SyncModel::ModelInfo`]: crate::SyncModel::ModelInfo
pub struct AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    // R: ReplayBufferBase + Sync + Send + 'static,
    R: ReplayBufferBase,
    R::PushedItem: Send + 'static,
{
    /// Where to save the trained model.
    model_dir: Option<String>,

    /// Interval of recording in training steps.
    record_interval: usize,

    /// Interval of evaluation in training steps.
    eval_interval: usize,

    /// The maximal number of training steps.
    max_train_steps: usize,

    /// Interval of saving the model in optimization steps.
    save_interval: usize,

    /// Interval of synchronizing model parameters in training steps.
    sync_interval: usize,

    /// The number of episodes for evaluation
    eval_episodes: usize,

    /// Receiver of pushed items.
    r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,

    /// If `false`, stops the actor threads.
    stop: Arc<Mutex<bool>>,

    /// Configuration of [Agent].
    agent_config: A::Config,

    /// Configuration of [Env]. Note that it is used only for evaluation, not for training.
    env_config: E::Config,

    /// Sender of model info.
    model_info_sender: Sender<(usize, A::ModelInfo)>,

    /// Configuration of replay buffer.
    replay_buffer_config: R::Config,

    phantom: PhantomData<(A, E, R)>,
}

impl<A, E, R> AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    // R: ReplayBufferBase + Sync + Send + 'static,
    R: ReplayBufferBase,
    R::PushedItem: Send + 'static,
{
    /// Creates [`AsyncTrainer`].
    pub fn build(
        config: &AsyncTrainerConfig,
        agent_config: &A::Config,
        env_config: &E::Config,
        replay_buffer_config: &R::Config,
        r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,
        model_info_sender: Sender<(usize, A::ModelInfo)>,
        stop: Arc<Mutex<bool>>,
    ) -> Self {
        Self {
            model_dir: config.model_dir.clone(),
            record_interval: config.record_interval,
            eval_interval: config.eval_interval,
            max_train_steps: config.max_train_steps,
            save_interval: config.save_interval,
            sync_interval: config.sync_interval,
            eval_episodes: config.eval_episodes,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            replay_buffer_config: replay_buffer_config.clone(),
            r_bulk_pushed_item,
            model_info_sender,
            stop,
            phantom: PhantomData,
        }
    }

    fn save_model(agent: &A, model_dir: String) {
        match agent.save(&model_dir) {
            Ok(()) => info!("Saved the model in {:?}", &model_dir),
            Err(_) => info!("Failed to save model."),
        }
    }

    fn evaluate(&mut self, agent: &mut A, env: &mut E) -> Result<f32> {
        agent.eval();

        let mut r_total = 0f32;

        for ix in 0..self.eval_episodes {
            let mut prev_obs = env.reset_with_index(ix)?;
            assert_eq!(prev_obs.len(), 1); // env must be non-vectorized

            loop {
                let act = agent.sample(&prev_obs);
                let (step, _) = env.step(&act);
                r_total += step.reward[0];
                if step.is_done[0] == 1 {
                    break;
                }
                prev_obs = step.obs;
            }
        }

        agent.train();

        Ok(r_total / self.eval_episodes as f32)
    }

    /// Do evaluation.
    #[inline(always)]
    fn eval(&mut self, agent: &mut A, env: &mut E, record: &mut Record, max_eval_reward: &mut f32) {
        let eval_reward = self.evaluate(agent, env).unwrap();
        record.insert("eval_reward", Scalar(eval_reward));

        // Save the best model up to the current iteration
        if eval_reward > *max_eval_reward {
            *max_eval_reward = eval_reward;
            let model_dir = self.model_dir.as_ref().unwrap().clone() + "/best";
            Self::save_model(agent, model_dir);
            info!("Saved the best model");
        }
    }

    /// Record.
    #[inline]
    fn record(
        &mut self,
        record: &mut Record,
        opt_steps_: &mut usize,
        samples: &mut usize,
        time: &mut SystemTime,
        samples_total: usize,
    ) {
        let duration = time.elapsed().unwrap().as_secs_f32();
        let ops = (*opt_steps_ as f32) / duration;
        let sps = (*samples as f32) / duration;
        let spo = (*samples as f32) / (*opt_steps_ as f32);
        record.insert("samples_total", Scalar(samples_total as _));
        record.insert("opt_steps_per_sec", Scalar(ops));
        record.insert("samples_per_sec", Scalar(sps));
        record.insert("samples_per_opt_steps", Scalar(spo));
        // info!("Collected samples per optimization step = {}", spo);

        // Reset counter
        *opt_steps_ = 0;
        *samples = 0;
        *time = SystemTime::now();
    }

    /// Flush record.
    #[inline]
    fn flush(&mut self, opt_steps: usize, mut record: Record, recorder: &mut impl Recorder) {
        record.insert("opt_steps", Scalar(opt_steps as _));
        recorder.write(record);
    }

    /// Save model.
    #[inline]
    fn save(&mut self, opt_steps: usize, agent: &A) {
        let model_dir =
            self.model_dir.as_ref().unwrap().clone() + format!("/{}", opt_steps).as_str();
        Self::save_model(agent, model_dir);
    }

    /// Sync model.
    #[inline]
    fn sync(&mut self, agent: &A) {
        let model_info = agent.model_info();
        // TODO: error handling
        self.model_info_sender.send(model_info).unwrap();
    }

    // /// Run a thread for replay buffer.
    // fn run_replay_buffer_thread(&self, buffer: Arc<Mutex<R>>) {
    //     let r = self.r_bulk_pushed_item.clone();
    //     let stop = self.stop.clone();

    //     std::thread::spawn(move || loop {
    //         let msg = r.recv().unwrap();
    //         {
    //             let mut buffer = buffer.lock().unwrap();
    //             buffer.push(msg.pushed_item);
    //         }
    //         if *stop.lock().unwrap() {
    //             break;
    //         }
    //         std::thread::sleep(std::time::Duration::from_millis(100));
    //     });
    // }

    /// Runs training loop.
    ///
    /// In the training loop, the following values will be pushed into the given recorder:
    ///
    /// * `samples_total` - Total number of samples pushed into the replay buffer.
    ///   Here, a "sample" is an item in [`ExperienceBufferBase::PushedItem`].
    /// * `opt_steps_per_sec` - The number of optimization steps per second.
    /// * `samples_per_sec` - The number of samples per second.
    /// * `samples_per_opt_steps` - The number of samples per optimization step.
    ///
    /// These values will typically be monitored with tensorboard.
    ///
    /// [`ExperienceBufferBase::PushedItem`]: border_core::ExperienceBufferBase::PushedItem
    pub fn train(&mut self, recorder: &mut impl Recorder, guard_init_env: Arc<Mutex<bool>>) -> AsyncTrainStat {
        // TODO: error handling
        let mut env = {
            let mut tmp = guard_init_env.lock().unwrap();
            *tmp = true;
            E::build(&self.env_config, 0).unwrap()
        };
        let mut agent = A::build(self.agent_config.clone());
        let mut buffer = R::build(&self.replay_buffer_config);
        
        // message reciever from ActorManager
        let msg_reciever = {
            let mut msg_reciever = MassageReciever::<R>::new();
            msg_reciever.run(self.r_bulk_pushed_item.clone());
            msg_reciever
        };

        // let buffer = Arc::new(Mutex::new(R::build(&self.replay_buffer_config)));
        agent.train();

        // self.run_replay_buffer_thread(buffer.clone());

        let mut max_eval_reward = f32::MIN;
        let mut opt_steps = 0;
        let mut opt_steps_ = 0;
        let mut samples = 0;
        let time_total = SystemTime::now();
        let mut samples_total = 0;
        let mut time = SystemTime::now();

        info!("Send model info first in AsyncTrainer");
        self.sync(&mut agent);

        info!("Starts training loop");
        loop {
            // Update replay buffer
            let msgs = msg_reciever.get_messages();
            println!("msgs.len(): {}", msgs.len());
            msgs.into_iter().for_each(|msg| {
                samples += msg.pushed_items.len();
                samples_total += msg.pushed_items.len();
                msg.pushed_items
                    .into_iter()
                    .for_each(|pushed_item| buffer.push(pushed_item).unwrap())
            });

            let record = agent.opt(&mut buffer);

            if let Some(mut record) = record {
                opt_steps += 1;
                opt_steps_ += 1;

                let do_eval = opt_steps % self.eval_interval == 0;
                let do_record = opt_steps % self.record_interval == 0;
                let do_flush = do_eval || do_record;
                let do_save = opt_steps % self.save_interval == 0;
                let do_sync = opt_steps % self.sync_interval == 0;

                if do_eval {
                    info!("Starts evaluation of the trained model");
                    self.eval(&mut agent, &mut env, &mut record, &mut max_eval_reward);
                }
                if do_record {
                    info!("Records training logs");
                    self.record(&mut record, &mut opt_steps_, &mut samples, &mut time, samples_total);
                }
                if do_flush {
                    info!("Flushes records");
                    self.flush(opt_steps, record, recorder);
                }
                if do_save {
                    info!("Saves the trained model");
                    self.save(opt_steps, &mut agent);
                }
                if opt_steps == self.max_train_steps {
                    // Flush channels
                    *self.stop.lock().unwrap() = true;
                    // let _: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
                    msg_reciever.stop_and_join();
                    self.sync(&agent);
                    break;
                }
                if do_sync {
                    info!("Sends the trained model info to ActorManager");
                    self.sync(&agent);
                }
            }
        }
        info!("Stopped training loop");

        let duration = time_total.elapsed().unwrap();
        let time_total = duration.as_secs_f32();
        let samples_per_sec = samples_total as f32 / time_total;
        let opt_per_sec = self.max_train_steps as f32 / time_total;
        AsyncTrainStat {
            samples_per_sec,
            duration,
            opt_per_sec,
        }
    }
}

struct MassageReciever<R>
where
    R: ReplayBufferBase,
    R::PushedItem: Send + 'static,
{
    msgs: Arc<Mutex<Vec<PushedItemMessage<R::PushedItem>>>>,
    stop: Arc<Mutex<bool>>,
    thread: Option<JoinHandle<()>>,
}

impl <R> MassageReciever<R>
where
    R: ReplayBufferBase,
    R::PushedItem: Send + 'static,
{
    fn new(
    ) -> Self {
        Self {
            msgs: Arc::new(Mutex::new(Vec::<PushedItemMessage<R::PushedItem>>::new())),
            stop: Arc::new(Mutex::new(false)),
            thread: None,
        }
    }

    fn run(
        &mut self,
        receiver: Receiver<PushedItemMessage<R::PushedItem>>,
    ) {
        println!("bbb");
        let msgs = self.msgs.clone();
        let stop = self.stop.clone();
        let handle = std::thread::spawn(move || {
            Self::recieve_message(msgs, receiver, stop);
        });
        self.thread = Some(handle);
    }

    fn recieve_message(
        msgs: Arc<Mutex<Vec<PushedItemMessage<R::PushedItem>>>>,
        receiver: Receiver<PushedItemMessage<R::PushedItem>>,
        stop: Arc<Mutex<bool>>,
    ) {
        println!("aaa");
        for msg in receiver.iter() {
            msgs.lock().unwrap().push(msg);
        }

        println!("ccc");

        let _: Vec<_> = receiver.try_iter().collect();
        info!("Stopped thread for message recieving");
    }

    fn get_messages(&self) -> Vec<PushedItemMessage<R::PushedItem>> {
        // println!("msgs.len(): {}", self.msgs.lock().unwrap().len());
        let mut msgs_ = Vec::new();
        std::mem::swap(&mut *self.msgs.lock().unwrap(), &mut msgs_);
        msgs_
    }

    fn stop_and_join(self) {
        *self.stop.lock().unwrap() = true;
        self.thread.unwrap().join().unwrap();
    }
}
