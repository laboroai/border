use crate::{AsyncTrainStat, AsyncTrainerConfig, PushedItemMessage, SyncModel};
use border_core::{
    record::{Record, RecordValue::Scalar, Recorder},
    Agent, Configurable, Env, Evaluator, ExperienceBufferBase, ReplayBufferBase,
};
use crossbeam_channel::{Receiver, Sender};
use log::info;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
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
/// * The [`Agent`] in [`AsyncTrainer`] (left) is trained with batches
///   of type [`ReplayBufferBase::Batch`], which are taken from the replay buffer.
/// * The model parameters of the [`Agent`] in [`AsyncTrainer`] are wrapped in
///   [`SyncModel::ModelInfo`] and periodically sent to the [`Agent`]s in [`Actor`]s.
///   [`Agent`] must implement [`SyncModel`] to synchronize the model parameters.
/// * In [`ActorManager`] (right), [`Actor`]s sample transitions, which have type
///   [`ReplayBufferBase::Item`], and push the transitions into
///   [`ReplayBufferProxy`].
/// * [`ReplayBufferProxy`] has a type parameter of [`ReplayBufferBase`] and the proxy accepts
///   [`ReplayBufferBase::Item`].
/// * The proxy sends the transitions into the replay buffer in the [`AsyncTrainer`].
///
/// [`ActorManager`]: crate::ActorManager
/// [`Actor`]: crate::Actor
/// [`ReplayBufferBase::Item`]: border_core::ReplayBufferBase::PushedItem
/// [`ReplayBufferBase::Batch`]: border_core::ReplayBufferBase::PushedBatch
/// [`ReplayBufferProxy`]: crate::ReplayBufferProxy
/// [`ReplayBufferBase`]: border_core::ReplayBufferBase
/// [`SyncModel::ModelInfo`]: crate::SyncModel::ModelInfo
/// [`Agent`]: border_core::Agent
pub struct AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + Configurable + SyncModel,
    E: Env,
    // R: ReplayBufferBase + Sync + Send + 'static,
    R: ExperienceBufferBase + ReplayBufferBase,
    R::Item: Send + 'static,
{
    /// Configuration of [`Env`]. Note that it is used only for evaluation, not for training.
    env_config: E::Config,

    /// Configuration of the replay buffer.
    replay_buffer_config: R::Config,

    /// Interval of recording computational cost in optimization steps.
    record_compute_cost_interval: usize,

    /// Interval of flushing records in optimization steps.
    flush_records_interval: usize,

    /// Interval of evaluation in training steps.
    eval_interval: usize,

    /// Interval of saving the model in optimization steps.
    save_interval: usize,

    /// The maximal number of optimization steps.
    max_opts: usize,

    /// Optimization steps for computing optimization steps per second.
    opt_steps_for_ops: usize,

    /// Timer for computing for optimization steps per second.
    timer_for_ops: Duration,

    /// Warmup period, for filling replay buffer, in environment steps
    warmup_period: usize,

    /// Interval of synchronizing model parameters in training steps.
    sync_interval: usize,

    /// Receiver of pushed items.
    r_bulk_pushed_item: Receiver<PushedItemMessage<R::Item>>,

    /// If `false`, stops the actor threads.
    stop: Arc<Mutex<bool>>,

    /// Configuration of [`Agent`].
    agent_config: A::Config,

    /// Sender of model info.
    model_info_sender: Sender<(usize, A::ModelInfo)>,

    phantom: PhantomData<(A, E, R)>,
}

impl<A, E, R> AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + Configurable + SyncModel + 'static,
    E: Env,
    // R: ReplayBufferBase + Sync + Send + 'static,
    R: ExperienceBufferBase + ReplayBufferBase,
    R::Item: Send + 'static,
{
    /// Creates [`AsyncTrainer`].
    pub fn build(
        config: &AsyncTrainerConfig,
        agent_config: &A::Config,
        env_config: &E::Config,
        replay_buffer_config: &R::Config,
        r_bulk_pushed_item: Receiver<PushedItemMessage<R::Item>>,
        model_info_sender: Sender<(usize, A::ModelInfo)>,
        stop: Arc<Mutex<bool>>,
    ) -> Self {
        Self {
            eval_interval: config.eval_interval,
            max_opts: config.max_opts,
            record_compute_cost_interval: config.record_compute_cost_interval,
            flush_records_interval: config.flush_record_interval,
            save_interval: config.save_interval,
            sync_interval: config.sync_interval,
            warmup_period: config.warmup_period,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            replay_buffer_config: replay_buffer_config.clone(),
            r_bulk_pushed_item,
            model_info_sender,
            stop,
            opt_steps_for_ops: 0,
            timer_for_ops: Duration::new(0, 0),
            phantom: PhantomData,
        }
    }

    /// Returns optimization steps per second, then reset the internal counter.
    fn opt_steps_per_sec(&mut self) -> f32 {
        let osps = 1000. * self.opt_steps_for_ops as f32 / (self.timer_for_ops.as_millis() as f32);
        self.opt_steps_for_ops = 0;
        self.timer_for_ops = Duration::new(0, 0);
        osps
    }

    #[inline]
    fn downcast_ref(agent: &Box<dyn Agent<E, R>>) -> &A {
        agent.deref().as_any_ref().downcast_ref::<A>().unwrap()
    }

    #[inline]
    fn downcast_mut(agent: &mut Box<dyn Agent<E, R>>) -> &mut A {
        agent.deref_mut().as_any_mut().downcast_mut::<A>().unwrap()
    }

    #[inline]
    fn train_step(&mut self, agent: &mut Box<dyn Agent<E, R>>, buffer: &mut R) -> Record {
        let timer = SystemTime::now();
        let record = agent.opt_with_record(buffer);
        self.opt_steps_for_ops += 1;
        self.timer_for_ops += timer.elapsed().unwrap();
        record
    }

    /// Synchronize model.
    #[inline]
    fn sync(&mut self, agent: &A) {
        let model_info = agent.model_info();
        // TODO: error handling
        self.model_info_sender.send(model_info).unwrap();
    }

    #[inline]
    fn update_replay_buffer(
        &mut self,
        buffer: &mut R,
        samples: &mut usize,
        samples_total: &mut usize,
    ) {
        let msgs: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
        msgs.into_iter().for_each(|msg| {
            *samples += msg.pushed_items.len();
            *samples_total += msg.pushed_items.len();
            msg.pushed_items
                .into_iter()
                .for_each(|pushed_item| buffer.push(pushed_item).unwrap())
        });
    }

    /// Runs training loop.
    ///
    /// In the training loop, the following values will be pushed into the given recorder:
    ///
    /// * `samples_total` - Total number of samples pushed into the replay buffer.
    ///   Here, a "sample" is an item in [`ExperienceBufferBase::Item`].
    /// * `opt_steps_per_sec` - The number of optimization steps per second.
    /// * `samples_per_sec` - The number of samples per second.
    /// * `samples_per_opt_steps` - The number of samples per optimization step.
    ///
    /// These values will typically be monitored with tensorboard.
    ///
    /// [`ExperienceBufferBase::Item`]: border_core::ExperienceBufferBase::Item
    pub fn train<D>(
        &mut self,
        recorder: &mut Box<dyn Recorder<E, R>>,
        evaluator: &mut D,
        guard_init_env: Arc<Mutex<bool>>,
    ) -> AsyncTrainStat
    where
        D: Evaluator<E>,
    {
        // TODO: error handling
        let _env = {
            let mut tmp = guard_init_env.lock().unwrap();
            *tmp = true;
            E::build(&self.env_config, 0).unwrap()
        };
        let mut agent: Box<dyn Agent<E, R>> = Box::new(A::build(self.agent_config.clone()));
        let mut buffer = R::build(&self.replay_buffer_config);
        agent.train();

        let mut max_eval_reward = f32::MIN;
        let mut opt_steps = 0;
        let mut samples = 0;
        let time_total = SystemTime::now();
        let mut samples_total = 0;

        info!("Send model info first in AsyncTrainer");
        self.sync(Self::downcast_ref(&agent));

        info!("Starts training loop");
        loop {
            self.update_replay_buffer(&mut buffer, &mut samples, &mut samples_total);

            if buffer.len() < self.warmup_period {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            let mut record = self.train_step(&mut agent, &mut buffer);
            opt_steps += 1;

            // Add stats wrt computation cost
            if opt_steps % self.record_compute_cost_interval == 0 {
                record.insert("opt_steps_per_sec", Scalar(self.opt_steps_per_sec()));
            }

            // Evaluation
            if opt_steps % self.eval_interval == 0 {
                info!("Starts evaluation of the trained model");
                agent.eval();
                let eval_reward = evaluator.evaluate(&mut agent).unwrap();
                let eval_reward_value = eval_reward.get_scalar_without_key();
                agent.train();
                record.merge_inplace(eval_reward);

                // Save the best model up to the current iteration
                if let Some(eval_reward) = eval_reward_value {
                    if eval_reward > max_eval_reward {
                        max_eval_reward = eval_reward;
                        recorder
                            .save_model("best".as_ref(), &mut agent)
                            .expect("Failed to save model");
                    }
                }
            }

            // Save the current model
            if (self.save_interval > 0) && (opt_steps % self.save_interval == 0) {
                recorder
                    .save_model(format!("{}", opt_steps).as_ref(), &mut agent)
                    .expect("Failed to save model");
            }

            // Finish the training loop
            if opt_steps == self.max_opts {
                // Flush channels
                *self.stop.lock().unwrap() = true;
                let _: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
                self.sync(Self::downcast_mut(&mut agent));
                break;
            }

            // Sync the current model
            if opt_steps % self.sync_interval == 0 {
                info!("Sends the trained model info to ActorManager");
                self.sync(Self::downcast_mut(&mut agent));
            }

            // Store record to the recorder
            if !record.is_empty() {
                recorder.store(record);
            }

            // Flush records
            if opt_steps % self.flush_records_interval == 0 {
                recorder.flush(opt_steps as _);
            }
        }
        info!("Stopped training loop");

        let duration = time_total.elapsed().unwrap();
        let time_total = duration.as_secs_f32();
        let samples_per_sec = samples_total as f32 / time_total;
        let opt_per_sec = self.max_opts as f32 / time_total;
        AsyncTrainStat {
            samples_per_sec,
            duration,
            opt_per_sec,
        }
    }
}
