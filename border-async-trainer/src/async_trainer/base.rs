use crate::{AsyncTrainerConfig, PushedItemMessage, SyncModel};
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
};

/// Manages asynchronous training loop in a single machine.
///
/// It will be used with [ActorManager](crate::ActorManager)
pub struct AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    R: ReplayBufferBase + Sync + Send + 'static,
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

    /// If `false`, stops the thread for replay buffer.
    stop: Arc<Mutex<bool>>,

    /// Configuration of [Agent].
    agent_config: A::Config,

    /// Configuration of [Env].
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
    R: ReplayBufferBase + Sync + Send + 'static,
    R::PushedItem: Send + 'static,
{
    /// Creates [AsyncTrainer].
    pub fn build(
        config: &AsyncTrainerConfig,
        agent_config: &A::Config,
        env_config: &E::Config,
        replay_buffer_config: &R::Config,
        r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,
        model_info_sender: Sender<(usize, A::ModelInfo)>,
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
            stop: Arc::new(Mutex::new(false)),
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

        for _ in 0..self.eval_episodes {
            let mut prev_obs = env.reset(None)?;
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
    fn record(&mut self, record: &mut Record, opt_steps_: &mut usize, time: &mut SystemTime) {
        let duration = time.elapsed().unwrap().as_secs_f32();
        let ops = (*opt_steps_ as f32) / duration;
        record.insert("Optimizations per second", Scalar(ops));

        // Reset counter
        *opt_steps_ = 0;
        *time = SystemTime::now();
    }

    /// Flush record.
    fn flush(&mut self, opt_steps: usize, mut record: Record, recorder: &mut impl Recorder) {
        record.insert("opt_steps", Scalar(opt_steps as _));
        recorder.write(record);
    }

    /// Save model.
    fn save(&mut self, opt_steps: usize, agent: &A) {
        let model_dir =
            self.model_dir.as_ref().unwrap().clone() + format!("/{}", opt_steps).as_str();
        Self::save_model(agent, model_dir);
    }

    /// Sync model.
    fn sync(&mut self, agent: &A) {
        let model_info = agent.model_info();
        // TODO: error handling
        self.model_info_sender.send(model_info).unwrap();
    }

    /// Run a thread for replay buffer.
    fn run_replay_buffer_thread(&self, buffer: Arc<Mutex<R>>) {
        let r = self.r_bulk_pushed_item.clone();
        let stop = self.stop.clone();

        std::thread::spawn(move || loop {
            let msg = r.recv().unwrap();
            {
                let mut buffer = buffer.lock().unwrap();
                buffer.push(msg.pushed_item);
            }
            if *stop.lock().unwrap() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        });
    }

    /// Runs training loop.
    pub fn train(&mut self, recorder: &mut impl Recorder, guard_init_env: Arc<Mutex<bool>>) {
        // TODO: error handling
        let mut env = {
            let mut tmp = guard_init_env.lock().unwrap();
            *tmp = true;
            E::build(&self.env_config, 0).unwrap()
        };
        let mut agent = A::build(self.agent_config.clone());
        let buffer = Arc::new(Mutex::new(R::build(&self.replay_buffer_config)));
        agent.train();

        self.run_replay_buffer_thread(buffer.clone());

        let mut max_eval_reward = f32::MIN;
        let mut opt_steps = 0;
        let mut opt_steps_ = 0;
        let mut time = SystemTime::now();

        info!("Send model info first in AsyncTrainer");
        self.sync(&mut agent);

        info!("Starts training loop");
        loop {
            let record = {
                let mut buffer = buffer.lock().unwrap();
                agent.opt(&mut buffer)
            };

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
                    self.record(&mut record, &mut opt_steps_, &mut time);
                }
                if do_flush {
                    info!("Flushes records");
                    self.flush(opt_steps, record, recorder);
                }
                if do_save {
                    info!("Saves the trained model");
                    self.save(opt_steps, &mut agent);
                }
                if do_sync {
                    info!("Sends the trained model info to ActorManager");
                    self.sync(&agent);
                }
                if opt_steps == self.max_train_steps {
                    break;
                }
            }
        }
    }
}
