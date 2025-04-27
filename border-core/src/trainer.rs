//! Training loop management and agent optimization.
//!
//! This module provides functionality for managing the training process of reinforcement
//! learning agents. It handles environment interactions, experience collection,
//! optimization steps, and evaluation.

mod config;
mod sampler;
use std::time::{Duration, SystemTime};

use crate::{
    record::{Record, RecordValue::Scalar, Recorder},
    Agent, Env, Evaluator, ExperienceBufferBase, ReplayBufferBase, StepProcessor,
};
use anyhow::Result;
pub use config::TrainerConfig;
use log::info;
pub use sampler::Sampler;

/// Manages the training loop and coordinates interactions between components.
///
/// The `Trainer` orchestrates the training process by managing:
///
/// * Environment interactions and experience collection
/// * Agent optimization and parameter updates
/// * Performance evaluation and model saving
/// * Training metrics recording
///
/// # Training Process
///
/// The training loop follows these steps:
///
/// 1. Initialize training components:
///    * Reset environment step counter (`env_steps = 0`)
///    * Reset optimization step counter (`opt_steps = 0`)
///    * Initialize performance monitoring
///
/// 2. Environment Interaction:
///    * Agent observes environment state
///    * Agent selects and executes action
///    * Environment transitions to new state
///    * Experience is collected and stored
///
/// 3. Optimization:
///    * At specified intervals (`opt_interval`):
///      * Sample experiences from replay buffer
///      * Update agent parameters
///      * Track optimization performance
///
/// 4. Evaluation and Recording:
///    * Periodically evaluate agent performance
///    * Record training metrics
///    * Save model checkpoints
///    * Monitor optimization speed
///
/// # Model Selection
///
/// During training, the best performing model is automatically saved based on evaluation rewards:
///
/// * At each evaluation interval (`eval_interval`), the agent's performance is evaluated
/// * The evaluation reward is obtained from `Record::get_scalar_without_key()`
/// * If the current evaluation reward exceeds the previous maximum reward:
///   * The model is saved as the "best" model
///   * The maximum reward is updated
/// * This ensures that the saved "best" model represents the agent's peak performance
///
/// # Configuration
///
/// Training behavior is controlled by various intervals and parameters:
///
/// * `opt_interval`: Steps between optimization updates
/// * `eval_interval`: Steps between performance evaluations
/// * `save_interval`: Steps between model checkpoints
/// * `warmup_period`: Initial steps before optimization begins
/// * `max_opts`: Maximum number of optimization steps
pub struct Trainer {
    /// Interval between optimization steps in environment steps.
    /// Ignored for offline training.
    opt_interval: usize,

    /// Interval for recording computational cost in optimization steps.
    record_compute_cost_interval: usize,

    /// Interval for recording agent information in optimization steps.
    record_agent_info_interval: usize,

    /// Interval for flushing records in optimization steps.
    flush_records_interval: usize,

    /// Interval for evaluation in optimization steps.
    eval_interval: usize,

    /// Interval for saving the model in optimization steps.
    save_interval: usize,

    /// Maximum number of optimization steps.
    max_opts: usize,

    /// Warmup period for filling replay buffer in environment steps.
    /// Ignored for offline training.
    warmup_period: usize,

    /// Counter for optimization steps used in performance calculation.
    opt_steps_for_ops: usize,

    /// Timer for measuring optimization performance.
    timer_for_ops: Duration,

    /// Maximum evaluation reward achieved.
    max_eval_reward: f32,

    /// Current environment step count.
    env_steps: usize,

    /// Current optimization step count.
    opt_steps: usize,
}

impl Trainer {
    /// Creates a new trainer with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the trainer
    ///
    /// # Returns
    ///
    /// A new `Trainer` instance with the specified configuration
    pub fn build(config: TrainerConfig) -> Self {
        Self {
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

    /// Calculates and returns the current optimization steps per second.
    ///
    /// This method resets the internal counters after calculation.
    ///
    /// # Returns
    ///
    /// The number of optimization steps performed per second
    fn opt_steps_per_sec(&mut self) -> f32 {
        let osps = 1000. * self.opt_steps_for_ops as f32 / (self.timer_for_ops.as_millis() as f32);
        self.opt_steps_for_ops = 0;
        self.timer_for_ops = Duration::new(0, 0);
        osps
    }

    /// Performs a single training step.
    ///
    /// This method:
    /// 1. Performs an environment step
    /// 2. Collects and stores the experience
    /// 3. Optionally performs an optimization step
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent being trained
    /// * `buffer` - The replay buffer storing experiences
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A record of the training step
    /// * A boolean indicating if an optimization step was performed
    ///
    /// # Errors
    ///
    /// Returns an error if the optimization step fails
    pub fn train_step<E, R>(
        &mut self,
        agent: &mut Box<dyn Agent<E, R>>,
        buffer: &mut R,
    ) -> Result<(Record, bool)>
    where
        E: Env,
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

    fn post_process<E, R, D>(
        &mut self,
        agent: &mut Box<dyn Agent<E, R>>,
        evaluator: &mut D,
        recorder: &mut Box<dyn Recorder<E, R>>,
        record: &mut Record,
        fps: f32,
    ) -> Result<()>
    where
        E: Env,
        R: ReplayBufferBase,
        D: Evaluator<E>,
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
            let eval_reward_value = eval_reward.get_scalar_without_key();
            agent.train();
            record.merge_inplace(eval_reward);

            // Save the best model up to the current iteration
            if let Some(eval_reward) = eval_reward_value {
                if eval_reward > self.max_eval_reward {
                    self.max_eval_reward = eval_reward;
                    recorder.save_model("best".as_ref(), agent)?;
                }
            }
        };

        // Save the current model
        if (self.save_interval > 0) && (self.opt_steps % self.save_interval == 0) {
            recorder.save_model(format!("{}", self.opt_steps).as_ref(), agent)?;
        }

        Ok(())
    }

    fn loop_step<E, R, D>(
        &mut self,
        agent: &mut Box<dyn Agent<E, R>>,
        buffer: &mut R,
        recorder: &mut Box<dyn Recorder<E, R>>,
        evaluator: &mut D,
        record: Record,
        fps: f32,
    ) -> Result<bool>
    where
        E: Env,
        R: ReplayBufferBase,
        D: Evaluator<E>,
    {
        // Performe optimization step(s)
        let (mut record, is_opt) = {
            let (r, is_opt) = self.train_step(agent, buffer)?;
            (record.merge(r), is_opt)
        };

        // Postprocessing after each training step
        if is_opt {
            self.post_process(agent, evaluator, recorder, &mut record, fps)?;

            // End loop
            if self.opt_steps == self.max_opts {
                return Ok(true);
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

        Ok(false)
    }

    /// Train the agent online.
    pub fn train<E, P, R, D>(
        &mut self,
        env: E,
        step_proc: P,
        agent: &mut Box<dyn Agent<E, R>>,
        buffer: &mut R,
        recorder: &mut Box<dyn Recorder<E, R>>,
        evaluator: &mut D,
    ) -> Result<()>
    where
        E: Env,
        P: StepProcessor<E>,
        R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
        D: Evaluator<E>,
    {
        let mut sampler = Sampler::new(env, step_proc);
        sampler.reset_fps_counter();
        agent.train();

        loop {
            let record = sampler.sample_and_push(agent, buffer)?;
            let fps = sampler.fps();
            self.env_steps += 1;

            if self.loop_step(agent, buffer, recorder, evaluator, record, fps)? {
                return Ok(());
            }
        }
    }

    /// Train the agent offline.
    pub fn train_offline<E, R, D>(
        &mut self,
        agent: &mut Box<dyn Agent<E, R>>,
        buffer: &mut R,
        recorder: &mut Box<dyn Recorder<E, R>>,
        evaluator: &mut D,
    ) -> Result<()>
    where
        E: Env,
        R: ReplayBufferBase,
        D: Evaluator<E>,
    {
        // Return empty record
        self.warmup_period = 0;
        self.opt_interval = 1;
        agent.train();
        let fps = 0f32;

        loop {
            let record = Record::empty();
            self.env_steps += 1;

            if self.loop_step(agent, buffer, recorder, evaluator, record, fps)? {
                return Ok(());
            }
        }
    }
}
