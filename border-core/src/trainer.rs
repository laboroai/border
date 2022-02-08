//! Train [Agent](crate::Agent).
mod config;
mod sampler;
use crate::{
    record::{Record, Recorder},
    Agent, Env, Obs, ReplayBufferBase, StepProcessorBase,
};
use anyhow::Result;
pub use config::TrainerConfig;
use log::info;
pub use sampler::SyncSampler;

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Manages training loop and related objects.
///
/// # Training loop
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
/// * The [`Step<E: Env>`] object will be processed by [`StepProcessorBase`] and
///   creates [`ReplayBufferBase::PushedItem`], typically representing a transition
///   `(o_t, a_t, o_t+1, r_t)`, where `o_t` is kept in the
///   [`StepProcessorBase`], while other items in the given [`Step<E: Env>`].
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
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// Configuration of the environment for training.
    pub env_config_train: E::Config,

    /// Configuration of the environment for evaluation.
    ///
    /// If `None`, `env_config_train` is used.
    pub env_config_eval: Option<E::Config>,

    /// Configuration of the transition producer.
    pub step_proc_config: P::Config,

    /// Configuration of the replay buffer.
    pub replay_buffer_config: R::Config,

    /// Where to save the trained model.
    pub model_dir: Option<String>,

    /// Interval of optimization in environment steps.
    pub opt_interval: usize,

    /// Interval of recording in optimization steps.
    pub record_interval: usize,

    /// Interval of evaluation in optimization steps.
    pub eval_interval: usize,

    /// Interval of saving the model in optimization steps.
    pub save_interval: usize,

    /// The maximal number of optimization steps.
    pub max_opts: usize,

    /// The number of episodes for evaluation.
    pub eval_episodes: usize,
}

impl<E, P, R> Trainer<E, P, R>
where
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// test.
    pub fn f() {}
    /// Constructs a trainer.
    pub fn build(
        config: TrainerConfig,
        env_config_train: E::Config,
        env_config_eval: Option<E::Config>,
        step_proc_config: P::Config,
        replay_buffer_config: R::Config,
    ) -> Self {
        Self {
            env_config_train,
            env_config_eval,
            step_proc_config,
            replay_buffer_config,
            model_dir: config.model_dir,
            opt_interval: config.opt_interval,
            record_interval: config.record_interval,
            eval_interval: config.eval_interval,
            save_interval: config.save_interval,
            max_opts: config.max_opts,
            eval_episodes: config.eval_episodes,
        }
    }

    fn save_model<A: Agent<E, R>>(agent: &A, model_dir: String) {
        match agent.save(&model_dir) {
            Ok(()) => info!("Saved the model in {:?}", &model_dir),
            Err(_) => info!("Failed to save model."),
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

    /// Run episodes with the given agent and returns the average of cumulative reward.
    fn evaluate<A>(&mut self, agent: &mut A) -> Result<f32>
    where
        A: Agent<E, R>,
    {
        agent.eval();

        let env_config = if self.env_config_eval.is_none() {
            &self.env_config_train
        } else {
            &self.env_config_eval.as_ref().unwrap()
        };

        let mut env = E::build(env_config, 0)?; // TODO use eval_env_config
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

    /// Performs a training step.
    pub fn train_step<A: Agent<E, R>>(
        &self,
        agent: &mut A,
        buffer: &mut R,
        sampler: &mut SyncSampler<E, P>,
        env_steps: &mut usize,
    ) -> Result<(Option<Record>, Option<std::time::Duration>)>
    where
        A: Agent<E, R>,
    {
        // Sample transition(s) and push it into the replay buffer
        let record_ = sampler.sample_and_push(agent, buffer)?;

        // Do optimization step
        *env_steps += 1;

        if *env_steps % self.opt_interval == 0 {
            let now = std::time::SystemTime::now();
            let record = agent.opt(buffer).map_or(None, |r| Some(record_.merge(r)));
            let time = now.elapsed()?;
            Ok((record, Some(time)))
        } else {
            Ok((None, None))
        }
    }

    /// Train the agent.
    pub fn train<A, S>(&mut self, agent: &mut A, recorder: &mut S) -> Result<()>
    where
        A: Agent<E, R>,
        S: Recorder,
    {
        let env = E::build(&self.env_config_train, 0)?;
        let producer = P::build(&self.step_proc_config);
        let mut buffer = R::build(&self.replay_buffer_config);
        let mut sampler = SyncSampler::new(env, producer);
        let mut max_eval_reward = f32::MIN;
        let mut env_steps: usize = 0;
        let mut opt_steps: usize = 0;
        let mut opt_time: f32 = 0.;
        let mut opt_steps_ops: usize = 0; // optimizations per second
        sampler.reset();
        agent.train();

        loop {
            let (record, time) =
                self.train_step(agent, &mut buffer, &mut sampler, &mut env_steps)?;

            // Postprocessing after each training step
            if let Some(mut record) = record {
                use crate::record::RecordValue::Scalar;

                opt_steps += 1;

                // For calculating optimizations per seconds
                if let Some(time) = time {
                    opt_steps_ops += 1;
                    opt_time += time.as_millis() as f32;
                }

                let do_eval = opt_steps % self.eval_interval == 0;
                let do_rec = opt_steps % self.record_interval == 0;

                // Do evaluation
                if do_eval {
                    let eval_reward = self.evaluate(agent)?;
                    record.insert("eval_reward", Scalar(eval_reward));

                    // Save the best model up to the current iteration
                    if eval_reward > max_eval_reward {
                        max_eval_reward = eval_reward;
                        let model_dir = self.model_dir.as_ref().unwrap().clone();
                        Self::save_best_model(agent, model_dir)
                    }
                };

                // Record
                if do_rec {
                    record.insert("env_steps", Scalar(env_steps as f32));
                    record.insert("fps", Scalar(sampler.fps()));
                    sampler.reset();
                    let ops = opt_steps_ops as f32 * 1000. / opt_time;
                    record.insert("ops", Scalar(ops));
                    opt_steps_ops = 0;
                    opt_time = 0.;
                }

                // Flush record to the recorder
                if do_eval || do_rec {
                    record.insert("opt_steps", Scalar(opt_steps as _));
                    recorder.write(record);
                }

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
        }

        Ok(())
    }
}
