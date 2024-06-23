//! Train [`Agent`].
mod config;
mod sampler;
use std::time::{Duration, SystemTime};

use crate::{
    record::{AggregateRecorder, Record, RecordValue::Scalar},
    Agent, Env, Evaluator, ExperienceBufferBase, ReplayBufferBase, StepProcessor,
};
use anyhow::Result;
pub use config::TrainerConfig;
use log::info;
pub use sampler::Sampler;

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Manages training loop and related objects.
///
/// # Training loop
///
/// Training loop looks like following:
///
/// 0. Given an agent implementing [`Agent`]  and a recorder implementing [`Recorder`].
/// 1. Initialize the objects used in the training loop, involving instances of [`Env`],
///    [`StepProcessor`], [`Sampler`].
///    * Reset a counter of the environment steps: `env_steps = 0`
///    * Reset a counter of the optimization steps: `opt_steps = 0`
///    * Reset objects for computing optimization steps per sec (OSPS):
///        * A timer `timer = SystemTime::now()`.
///        * A counter `opt_steps_ops = 0`
/// 2. Reset [`Env`].
/// 3. Do an environment step and push a transition to the replaybuffer, implementing
///    [`ReplayBufferBase`].
/// 4. `env_steps += 1`
/// 5. If `env_steps % opt_interval == 0`:
///     1. Do an optimization step for the agent with transition batches
///        sampled from the replay buffer.
///         * NOTE: Here, the agent can skip an optimization step because of some reason,
///           for example, during a warmup period for the replay buffer.
///           In this case, the following steps are skipped as well.
///     2. `opt_steps += 1, opt_steps_ops += 1`
///     3. If `opt_steps % eval_interval == 0`:
///         * Do an evaluation of the agent and add the evaluation result to the as
///           `"eval_reward"`.
///         * Reset `timer` and `opt_steps_ops`.
///         * If the evaluation result is the best, agent's model parameters are saved
///           in directory `(model_dir)/best`.
///     4. If `opt_steps % record_interval == 0`, compute OSPS as
///        `opt_steps_ops / timer.elapsed()?.as_secs_f32()` and add it to the
///        recorder as `"opt_steps_per_sec"`.
///     5. If `opt_steps % save_interval == 0`, agent's model parameters are saved
///        in directory `(model_dir)/(opt_steps)`.
///     6. If `opt_steps == max_opts`, finish training loop.
/// 6. Back to step 3.
///
/// # Interaction of objects
///
/// In [`Trainer::train()`] method, objects interact as shown below:
///
/// ```mermaid
/// graph LR
///     A[Agent]-->|Env::Act|B[Env]
///     B -->|Env::Obs|A
///     B -->|"Step&lt;E: Env&gt;"|C[StepProcessor]
///     C -->|ReplayBufferBase::PushedItem|D[ReplayBufferBase]
///     D -->|BatchBase|A
/// ```
///
/// * First, [`Agent`] emits an [`Env::Act`] `a_t` based on [`Env::Obs`] `o_t` received from
///   [`Env`]. Given `a_t`, [`Env`] changes its state and creates the observation at the
///   next step, `o_t+1`. This step of interaction between [`Agent`] and [`Env`] is
///   referred to as an *environment step*.
/// * Next, [`Step<E: Env>`] will be created with the next observation `o_t+1`,
///   reward `r_t`, and `a_t`.
/// * The [`Step<E: Env>`] object will be processed by [`StepProcessor`] and
///   creates [`ReplayBufferBase::Item`], typically representing a transition
///   `(o_t, a_t, o_t+1, r_t)`, where `o_t` is kept in the
///   [`StepProcessor`], while other items in the given [`Step<E: Env>`].
/// * Finally, the transitions pushed to the [`ReplayBufferBase`] will be used to create
///   batches, each of which implementing [`BatchBase`]. These batches will be used in
///   *optimization step*s, where the agent updates its parameters using sampled
///   experiencesp in batches.
///
/// [`Trainer::train()`]: Trainer::train
/// [`Act`]: crate::Act
/// [`BatchBase`]: crate::BatchBase
/// [`Step<E: Env>`]: crate::Step
pub struct Trainer {
    /// Where to save the trained model.
    model_dir: Option<String>,

    /// Interval of optimization in environment steps.
    /// This is ignored for offline training.
    opt_interval: usize,

    /// Interval of recording computational cost in optimization steps.
    record_compute_cost_interval: usize,

    /// Interval of recording agent information in optimization steps.
    record_agent_info_interval: usize,

    /// Interval of flushing records in optimization steps.
    flush_records_interval: usize,

    /// Interval of evaluation in optimization steps.
    eval_interval: usize,

    /// Interval of saving the model in optimization steps.
    save_interval: usize,

    /// The maximal number of optimization steps.
    max_opts: usize,

    /// Optimization steps for computing optimization steps per second.
    opt_steps_for_ops: usize,

    /// Timer for computing for optimization steps per second.
    timer_for_ops: Duration,

    /// Warmup period, for filling replay buffer, in environment steps.
    /// This is ignored for offline training.
    warmup_period: usize,

    /// Max value of evaluation reward.
    max_eval_reward: f32,

    /// Environment steps during online training.
    env_steps: usize,

    /// Optimization steps during training.
    opt_steps: usize,
}

impl Trainer {
    /// Constructs a trainer.
    pub fn build(config: TrainerConfig) -> Self {
        Self {
            model_dir: config.model_dir,
            opt_interval: config.opt_interval,
            record_compute_cost_interval: config.record_compute_cost_interval,
            record_agent_info_interval: config.record_agent_info_interval,
            flush_records_interval: config.flush_record_interval,
            eval_interval: config.eval_interval,
            save_interval: config.save_interval,
            max_opts: config.max_opts,
            warmup_period: config.warmup_period,
            opt_steps_for_ops: 0,
            timer_for_ops: Duration::new(0, 0),
            max_eval_reward: f32::MIN,
            env_steps: 0,
            opt_steps: 0,
        }
    }

    fn save_model<E, A, R>(agent: &A, model_dir: String)
    where
        E: Env,
        A: Agent<E, R>,
        R: ReplayBufferBase,
    {
        match agent.save(&model_dir) {
            Ok(()) => info!("Saved the model in {:?}.", &model_dir),
            Err(_) => info!("Failed to save model in {:?}.", &model_dir),
        }
    }

    fn save_best_model<E, A, R>(agent: &A, model_dir: String)
    where
        E: Env,
        A: Agent<E, R>,
        R: ReplayBufferBase,
    {
        let model_dir = model_dir + "/best";
        Self::save_model(agent, model_dir);
    }

    fn save_model_with_steps<E, A, R>(agent: &A, model_dir: String, steps: usize)
    where
        E: Env,
        A: Agent<E, R>,
        R: ReplayBufferBase,
    {
        let model_dir = model_dir + format!("/{}", steps).as_str();
        Self::save_model(agent, model_dir);
    }

    /// Returns optimization steps per second, then reset the internal counter.
    fn opt_steps_per_sec(&mut self) -> f32 {
        let osps = 1000. * self.opt_steps_for_ops as f32 / (self.timer_for_ops.as_millis() as f32);
        self.opt_steps_for_ops = 0;
        self.timer_for_ops = Duration::new(0, 0);
        osps
    }

    /// Performs a training step.
    ///
    /// First, it performes an environment step once and pushes a transition
    /// into the given buffer with [`Sampler`]. Then, if the number of environment steps
    /// reaches the optimization interval `opt_interval`, performes an optimization
    /// step.
    ///
    /// The second return value in the tuple is if an optimization step is done (`true`).
    // pub fn train_step<E, A, P, R>(
    pub fn train_step<E, A, R>(&mut self, agent: &mut A, buffer: &mut R) -> Result<(Record, bool)>
    where
        E: Env,
        A: Agent<E, R>,
        R: ReplayBufferBase,
    {
        if self.env_steps < self.warmup_period {
            Ok((Record::empty(), false))
        } else if self.env_steps % self.opt_interval != 0 {
            // skip optimization step
            Ok((Record::empty(), false))
        } else if (self.opt_steps + 1) % self.record_agent_info_interval == 0 {
            // Do optimization step with record
            let timer = SystemTime::now();
            let record_agent = agent.opt_with_record(buffer);
            self.opt_steps += 1;
            self.timer_for_ops += timer.elapsed()?;
            self.opt_steps_for_ops += 1;
            Ok((record_agent, true))
        } else {
            // Do optimization step without record
            let timer = SystemTime::now();
            agent.opt(buffer);
            self.opt_steps += 1;
            self.timer_for_ops += timer.elapsed()?;
            self.opt_steps_for_ops += 1;
            Ok((Record::empty(), true))
        }
    }

    fn post_process<E, A, R, D>(
        &mut self,
        agent: &mut A,
        evaluator: &mut D,
        record: &mut Record,
        fps: f32,
    ) -> Result<()>
    where
        E: Env,
        A: Agent<E, R>,
        R: ReplayBufferBase,
        D: Evaluator<E, A>,
    {
        // Add stats wrt computation cost
        if self.opt_steps % self.record_compute_cost_interval == 0 {
            record.insert("fps", Scalar(fps));
            record.insert("opt_steps_per_sec", Scalar(self.opt_steps_per_sec()));
        }

        // Evaluation
        if self.opt_steps % self.eval_interval == 0 {
            info!("Starts evaluation of the trained model");
            agent.eval();
            let eval_reward = evaluator.evaluate(agent)?;
            agent.train();
            record.insert("eval_reward", Scalar(eval_reward));

            // Save the best model up to the current iteration
            if eval_reward > self.max_eval_reward {
                self.max_eval_reward = eval_reward;
                let model_dir = self.model_dir.as_ref().unwrap().clone();
                Self::save_best_model(agent, model_dir)
            }
        };

        // Save the current model
        if (self.save_interval > 0) && (self.opt_steps % self.save_interval == 0) {
            let model_dir = self.model_dir.as_ref().unwrap().clone();
            Self::save_model_with_steps(agent, model_dir, self.opt_steps);
        }

        Ok(())
    }

    fn _train<E, A, R, D>(
        &mut self,
        agent: &mut A,
        buffer: &mut R,
        recorder: &mut Box<dyn AggregateRecorder>,
        evaluator: &mut D,
        mut f: impl FnMut(&mut A, &mut R, &mut usize) -> Result<(Record, f32)>,
    ) -> Result<()>
    where
        E: Env,
        A: Agent<E, R>,
        R: ExperienceBufferBase + ReplayBufferBase,
        D: Evaluator<E, A>,
    {
        loop {
            // Sample transition and push it into the replay buffer
            let (record, fps): (Record, f32) = f(agent, buffer, &mut self.env_steps)?;

            // Performe optimization step(s)
            let (mut record, is_opt) = {
                let (r, is_opt) = self.train_step(agent, buffer)?;
                (record.merge(r), is_opt)
            };

            // Postprocessing after each training step
            if is_opt {
                self.post_process(agent, evaluator, &mut record, fps)?;

                // End loop
                if self.opt_steps == self.max_opts {
                    break;
                }
            }

            // Store record to the recorder
            if !record.is_empty() {
                recorder.store(record);
            }

            // Flush records
            if is_opt && ((self.opt_steps - 1) % self.flush_records_interval == 0) {
                recorder.flush(self.opt_steps as _);
            }
        }

        Ok(())
    }

    /// Train the agent in online.
    pub fn train<E, A, P, R, D>(
        &mut self,
        env: E,
        step_proc: P,
        agent: &mut A,
        buffer: &mut R,
        recorder: &mut Box<dyn AggregateRecorder>,
        evaluator: &mut D,
    ) -> Result<()>
    where
        E: Env,
        A: Agent<E, R>,
        P: StepProcessor<E>,
        R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
        D: Evaluator<E, A>,
    {
        // Sample transition, push it to a buffer, return FPS
        let f = {
            let mut sampler = Sampler::new(env, step_proc);
            sampler.reset_fps_counter();
            move |agent: &mut A, buffer: &mut R, env_steps: &mut usize| -> Result<(Record, f32)> {
                let record = sampler.sample_and_push(agent, buffer)?;
                let fps = sampler.fps();
                *env_steps += 1;
                Ok((record, fps))
            }
        };
        agent.train();
        self._train(agent, buffer, recorder, evaluator, f)
    }

    /// Train the agent in offline.
    pub fn train_offline<E, A, P, R, D>(
        &mut self,
        agent: &mut A,
        buffer: &mut R,
        recorder: &mut Box<dyn AggregateRecorder>,
        evaluator: &mut D,
    ) -> Result<()>
    where
        E: Env,
        A: Agent<E, R>,
        P: StepProcessor<E>,
        R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
        D: Evaluator<E, A>,
    {
        // Return empty record
        let f = |_: &mut A, _: &mut R, _: &mut usize| -> Result<(Record, f32)> {
            Ok((Record::empty(), 0f32))
        };
        self.warmup_period = 0;
        self.opt_interval = 1;
        agent.train();
        self._train(agent, buffer, recorder, evaluator, f)
    }
}
