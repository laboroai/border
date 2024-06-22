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
pub struct Trainer<E, P, R>
where
    E: Env,
    P: StepProcessor<E>,
    R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
{
    /// Configuration of the environment for training.
    env_config_train: E::Config,

    /// Configuration of the replay buffer.
    replay_buffer_config: R::Config,

    /// Where to save the trained model.
    model_dir: Option<String>,

    /// Interval of optimization in environment steps.
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

    /// Configuration of the transition producer.
    step_proc_config: P::Config,

    /// Warmup period, for filling replay buffer, in environment steps
    warmup_period: usize,
}

impl<E, P, R> Trainer<E, P, R>
where
    E: Env,
    P: StepProcessor<E>,
    R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
{
    /// Constructs a trainer.
    pub fn build(
        config: TrainerConfig,
        env_config_train: E::Config,
        step_proc_config: P::Config,
        replay_buffer_config: R::Config,
    ) -> Self {
        Self {
            env_config_train,
            step_proc_config,
            replay_buffer_config,
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
        }
    }

    fn save_model<A: Agent<E, R>>(agent: &A, model_dir: String) {
        match agent.save(&model_dir) {
            Ok(()) => info!("Saved the model in {:?}.", &model_dir),
            Err(_) => info!("Failed to save model in {:?}.", &model_dir),
        }
    }

    fn save_best_model<A: Agent<E, R>>(agent: &A, model_dir: String) {
        let model_dir = model_dir + "/best";
        Self::save_model(agent, model_dir);
    }

    fn save_model_with_steps<A: Agent<E, R>>(agent: &A, model_dir: String, steps: usize) {
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
    pub fn train_step<A: Agent<E, R>>(
        &mut self,
        agent: &mut A,
        buffer: &mut R,
        sampler: &mut Sampler<E, P>,
        env_steps: &mut usize,
        opt_steps: &mut usize,
    ) -> Result<(Record, bool)>
    where
        A: Agent<E, R>,
    {
        // Sample transition and push it into the replay buffer
        let mut record = sampler.sample_and_push(agent, buffer)?;
        *env_steps += 1;

        // Optimization step
        if *env_steps < self.warmup_period {
            Ok((record, false))
        } else if *env_steps % self.opt_interval != 0 {
            // skip optimization step
            Ok((record, false))
        } else if (*opt_steps + 1) % self.record_agent_info_interval == 0 {
            // Do optimization step with record
            let timer = SystemTime::now();
            let record_agent = agent.opt_with_record(buffer);
            *opt_steps += 1;
            self.timer_for_ops += timer.elapsed()?;
            self.opt_steps_for_ops += 1;
            record = record.merge(record_agent);
            Ok((record, true))
        } else {
            // Do optimization step without record
            let timer = SystemTime::now();
            agent.opt(buffer);
            *opt_steps += 1;
            self.timer_for_ops += timer.elapsed()?;
            self.opt_steps_for_ops += 1;
            Ok((record, true))
        }
    }

    /// Train the agent.
    pub fn train<A, D>(
        &mut self,
        agent: &mut A,
        recorder: &mut Box<dyn AggregateRecorder>,
        evaluator: &mut D,
    ) -> Result<()>
    where
        A: Agent<E, R>,
        D: Evaluator<E, A>,
    {
        let env = E::build(&self.env_config_train, 0)?;
        let producer = P::build(&self.step_proc_config);
        let mut buffer = R::build(&self.replay_buffer_config);
        let mut sampler = Sampler::new(env, producer);
        let mut max_eval_reward = f32::MIN;
        let mut env_steps: usize = 0;
        let mut opt_steps: usize = 0;
        sampler.reset_fps_counter();
        agent.train();

        loop {
            let (mut record, is_opt) = self.train_step(
                agent,
                &mut buffer,
                &mut sampler,
                &mut env_steps,
                &mut opt_steps,
            )?;

            // Postprocessing after each training step
            if is_opt {
                // Add stats wrt computation cost
                if opt_steps % self.record_compute_cost_interval == 0 {
                    record.insert("fps", Scalar(sampler.fps()));
                    record.insert("opt_steps_per_sec", Scalar(self.opt_steps_per_sec()));
                }

                // Evaluation
                if opt_steps % self.eval_interval == 0 {
                    info!("Starts evaluation of the trained model");
                    agent.eval();
                    let eval_reward = evaluator.evaluate(agent)?;
                    agent.train();
                    record.insert("eval_reward", Scalar(eval_reward));

                    // Save the best model up to the current iteration
                    if eval_reward > max_eval_reward {
                        max_eval_reward = eval_reward;
                        let model_dir = self.model_dir.as_ref().unwrap().clone();
                        Self::save_best_model(agent, model_dir)
                    }
                };

                // Save the current model
                if (self.save_interval > 0) && (opt_steps % self.save_interval == 0) {
                    let model_dir = self.model_dir.as_ref().unwrap().clone();
                    Self::save_model_with_steps(agent, model_dir, opt_steps);
                }

                // End loop
                if opt_steps == self.max_opts {
                    break;
                }
            }

            // Store record to the recorder
            if !record.is_empty() {
                recorder.store(record);
            }

            // Flush records
            if is_opt && ((opt_steps - 1) % self.flush_records_interval == 0) {
                recorder.flush(opt_steps as _);
            }
        }

        Ok(())
    }
}
