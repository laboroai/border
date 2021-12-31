use crate::{AsyncTrainerConfig, PushedItemMessage};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue::Scalar, Recorder},
    Agent, Env, ReplayBufferBase,
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
    A: Agent<E, R>,
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

    /// Receiver of pushed items.
    r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,

    /// If `false`, stops the thread for replay buffer.
    stop: Arc<Mutex<bool>>,

    phantom: PhantomData<(A, E, R)>,
}

impl<A, E, R> AsyncTrainer<A, E, R>
where
    A: Agent<E, R>,
    E: Env,
    R: ReplayBufferBase + Sync + Send + 'static,
    R::PushedItem: Send + 'static,
{
    /// Creates [AsyncTrainer].
    pub fn build(
        config: AsyncTrainerConfig,
        r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,
    ) -> Self {
        Self {
            model_dir: config.model_dir,
            record_interval: config.record_interval,
            eval_interval: config.eval_interval,
            max_train_steps: config.max_train_steps,
            save_interval: config.save_interval,
            sync_interval: config.sync_interval,
            r_bulk_pushed_item,
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

    fn evaluate(&mut self, agent: &mut A) -> Result<f32> {
        unimplemented!();
    }

    /// Do evaluation.
    #[inline(always)]
    fn eval(&mut self, agent: &mut A, record: &mut Record, max_eval_reward: &mut f32) {
        let eval_reward = self.evaluate(agent).unwrap();
        record.insert("eval_reward", Scalar(eval_reward));

        // Save the best model up to the current iteration
        if eval_reward > *max_eval_reward {
            *max_eval_reward = eval_reward;
            let model_dir = self.model_dir.as_ref().unwrap().clone() + "/best";
            Self::save_model(agent, model_dir);
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
    fn sync(&mut self, _agent: &A) {
        unimplemented!();
    }

    /// Run a thread for replay buffer.
    fn run_replay_buffer_thread(&self, buffer: Arc<Mutex<R>>) {
        let r = self.r_bulk_pushed_item.clone();
        let stop = self.stop.clone();

        std::thread::spawn(move || {
            loop {
                let msg = r.recv().unwrap();
                {
                    let mut buffer = buffer.lock().unwrap();
                    buffer.push(msg.pushed_item);
                }
                if *stop.lock().unwrap() {
                    break;
                }
            }
        });
    }

    /// Runs training loop.
    pub fn train(&mut self, agent: &mut A, buffer: Arc<Mutex<R>>, recorder: &mut impl Recorder) {
        self.run_replay_buffer_thread(buffer.clone());

        let mut max_eval_reward = f32::MIN;
        let mut opt_steps = 0;
        let mut opt_steps_ = 0;
        let mut time = SystemTime::now();

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
                    self.eval(agent, &mut record, &mut max_eval_reward);
                }
                if do_record {
                    self.record(&mut record, &mut opt_steps_, &mut time);
                }
                if do_flush {
                    self.flush(opt_steps, record, recorder);
                }
                if do_save {
                    self.save(opt_steps, agent);
                }
                if do_sync {
                    self.sync(agent);
                }
                if opt_steps == self.max_train_steps {
                    break;
                }
            }
        }
    }
}
