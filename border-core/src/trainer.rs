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
/// ```mermaid
/// flowchart TB
///   Env(Env) -- Env::step --> Step([Step])
///   Step([Step]) -- TransitionProducer::transition --> Transition([Transition])
///   Transition([Transition]) -- ReplayBufferBase::push --> ReplayBufferBase(ReplayBufferBase)
///   ReplayBufferBase(ReplayBufferBase) -- Agent::opt --> RefReplayBufferBase(&ReplayBufferBase)
///   subgraph Agent
///     RefReplayBufferBase(&ReplayBufferBase) -- ReplayBufferBase::sample --> Batch([Batch])
///   end
///
/// ```
///
pub struct Trainer<E, P, R>
where
    E: Env,
    P: StepProcessorBase<E>,
    R: ReplayBufferBase<PushedItem = P::Output>,
{
    /// Configuration of the environment.
    pub env_config: E::Config,

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
    /// Constructs a trainer.
    pub fn build(config: TrainerConfig, env_config: E::Config, step_proc_config: P::Config, replay_buffer_config: R::Config) -> Self {
        Self {
            env_config,
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

        let mut env = E::build(&self.env_config, 0)?; // TODO use eval_env_config
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
        sampler.sample_and_push(agent, buffer)?;

        // Do optimization step
        *env_steps += 1;

        if *env_steps % self.opt_interval == 0 {
            let now = std::time::SystemTime::now();
            let record = agent.opt(buffer);
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
        let env = E::build(&self.env_config, 0)?;
        let producer = P::build(&self.step_proc_config);
        let mut buffer = R::build(&self.replay_buffer_config);
        let mut sampler = SyncSampler::new(env, producer);
        let mut max_eval_reward = f32::MIN;
        let mut env_steps: usize = 0;
        let mut opt_steps: usize = 0;
        let mut opt_time: f32 = 0.;
        let mut opt_steps_ops: usize = 0; // optimizations per second
        sampler.reset();

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

        // let sampler = SyncSampler::new(&agent, )

        // let mut count_steps_local = 0;
        // let mut now = std::time::SystemTime::now();
        // let mut max_eval_reward = std::f32::MIN;

        // let obs = self.env.reset(None).unwrap();
        // self.agent.push_obs(&obs);
        // self.obs_prev.replace(Some(obs));
        // self.agent.train(); // set to training mode

        // loop {
        //     let mut over_eval_threshold = false;

        //     // For resetted environments, elements in obs_prev are updated with env.reset().
        //     // After the update, obs_prev will have o_t+1 without reset, or o_0 with reset.
        //     // See `sample()` in `util.rs`.
        //     let (step, record_env) = sample(&mut self.env, &mut self.agent, &self.obs_prev);
        //     self.count_steps += 1;
        //     count_steps_local += 1;

        //     // agent.observe() internally creates transisions, i.e., (o_t, a_t, o_t+1, r_t+1).
        //     // For o_t, the previous observation stored in the agent is used.
        //     let option_record = self.agent.observe(step);

        //     // The previous observations in the agent are updated with obs_prev.
        //     // These are o_t+1 (without reset) or o_0 (with reset).
        //     // In the next iteration of the loop, o_t+1 will be treated as the previous observation
        //     // in the next training step executed in agent.observation().
        //     self.agent
        //         .push_obs(&self.obs_prev.borrow().as_ref().unwrap());

        //     if let Some(mut record) = option_record {
        //         use RecordValue::{DateTime, Scalar};

        //         self.count_opts += 1;
        //         record.insert("n_steps", Scalar(self.count_steps as _));
        //         record.insert("n_opts", Scalar(self.count_opts as _));
        //         record.insert("datetime", DateTime(Local::now()));

        //         if self.count_opts % self.eval_interval == 0 {
        //             // Show FPS before evaluation
        //             let fps = match now.elapsed() {
        //                 Ok(elapsed) => {
        //                     Some(count_steps_local as f32 / elapsed.as_millis() as f32 * 1000.0)
        //                 }
        //                 Err(_) => None,
        //             };
        //             // Reset counter for getting FPS in training
        //             count_steps_local = 0;

        //             // The timer is used to measure the elapsed time for evaluation
        //             now = std::time::SystemTime::now();

        //             // Evaluation
        //             self.agent.eval();
        //             let rewards = eval(
        //                 &mut self.env_eval,
        //                 &mut self.agent,
        //                 self.n_episodes_per_eval,
        //             );
        //             let (mean, min, max) = Self::stats_eval_reward(&rewards);
        //             info!(
        //                 "Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
        //                 self.count_opts, mean, min, max
        //             );
        //             record.insert("mean_cum_eval_reward", Scalar(mean));

        //             if let Some(fps) = fps {
        //                 info!("{} FPS in training", fps);
        //             }

        //             match now.elapsed() {
        //                 Ok(elapsed) => {
        //                     info!("{} sec. in evaluation", elapsed.as_millis() as f32 / 1000.0);
        //                 }
        //                 Err(_) => {
        //                     info!("An error occured when getting time")
        //                 }
        //             }

        //             // The timer is used to measure the elapsed time for training
        //             now = std::time::SystemTime::now();

        //             // Save the trained model
        //             if self.model_dir != None && mean > max_eval_reward {
        //                 if let Some(model_dir) = self.model_dir.clone() {
        //                     max_eval_reward = mean;
        //                     match self.agent.save(&model_dir) {
        //                         Ok(()) => info!("Saved the model in {:?}", &model_dir),
        //                         Err(_) => info!("Failed to save model."),
        //                     }
        //                 }
        //             }

        //             // Set the agent in evaluation mode
        //             self.agent.train();

        //             // If the evaluation reward exceeds the threshold
        //             if let Some(th) = self.eval_threshold {
        //                 over_eval_threshold = mean >= th;
        //             }

        //             let record = record.merge(record_env);
        //             recorder.write(record);
        //         }
        //     }

        //     if self.count_opts >= self.max_opts || over_eval_threshold {
        //         break;
        //     }
        // }
    }
}

// use anyhow::Result;
// use chrono::Local;
// use log::info;
// use serde::{Deserialize, Serialize};
// use std::{
//     cell::RefCell,
//     fs::File,
//     io::{BufReader, Write},
//     path::Path,
// };

//     ///Constructs a trainer.
//     pub fn build<E, A>(self, env: E, env_eval: E, agent: A) -> Trainer<E, A>
//     where
//         E: Env,
//         A: Agent<E>,
//     {
//         Trainer {
//             env,
//             env_eval,
//             agent,
//             obs_prev: RefCell::new(None),
//             max_opts: self.max_opts,
//             eval_interval: self.eval_interval,
//             n_episodes_per_eval: self.n_episodes_per_eval,
//             eval_threshold: self.eval_threshold,
//             model_dir: self.model_dir,
//             count_opts: 0,
//             count_steps: 0,
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tempdir::TempDir;

//     #[test]
//     fn test_serde_trainer_builder() -> Result<()> {
//         let builder = TrainerBuilder::default()
//             .max_opts(100)
//             .eval_interval(10000)
//             .n_episodes_per_eval(5)
//             .model_dir("some/directory");

//         let dir = TempDir::new("trainer_builder")?;
//         let path = dir.path().join("trainer_builder.yaml");
//         println!("{:?}", path);

//         builder.save(&path)?;
//         let builder_ = TrainerBuilder::load(&path)?;
//         assert_eq!(builder, builder_);
//         // let yaml = serde_yaml::to_string(&trainer)?;
//         // println!("{}", yaml);
//         // assert_eq!(
//         //     yaml,
//         //     "---\n\
//         //      max_opts: 100\n\
//         //      eval_interval: 10000\n\
//         //      n_episodes_per_eval: 5\n\
//         //      eval_threshold: ~\n\
//         //      model_dir: some/directory\n\
//         // "
//         // );
//         Ok(())
//     }
// }

// #[cfg_attr(doc, aquamarine::aquamarine)]
// /// Manages training process.
// ///
// /// ## Training loop
// ///
// /// For training an agent with standard RL algorithms in the library, the agent and environment
// /// interact as illustrated in the following diagram:
// ///
// /// ```mermaid
// /// flowchart TB
// ///     Trainer -. 0. Env::reset .-> Env
// ///     Env --> Obs
// ///     ObsPrev -- 3. Policy::sample --> Policy
// ///     Policy --> Act
// ///     Act -- 4. Env::step --> Env
// ///     Obs --> Step
// ///     Obs -- 1. RefCell::replace --> ObsPrev
// ///     Act --> Step
// ///     ObsPrev -- 2. Agent::push_obs --> ObsPrev'
// ///     Step -- 5. Agent::observe --> Transition
// ///
// ///     subgraph Agent
// ///         ObsPrev' --> Transition
// ///         ReplayBuffer -- 6. update policy parameters --- Policy
// ///         Transition --> ReplayBuffer
// ///     end
// /// ```
// ///
// /// 0. Call [`Env::reset`] for resetting the enbironment and getting an observation.
// /// An episode starts.
// /// 1. Call [`std::cell::RefCell::replace`] for placing the observation in `PrevObs`.
// /// 2. Call [`Agent::push_obs`] for placing the observation in `PrevObs'`.
// /// 3. Call [`Policy::sample`] for sampling an action from `Policy`.
// /// 4. Call [`Env::step`] for taking an action, getting a new observation, and creating [`Step`] object.
// /// 5. Call [`Agent::observe`] for updating the replay buffer with the new and previous observations.
// /// 6. Call some methods in the agent for updating policy parameters.
// /// 7. Back to 1.
// ///
// /// Actually, [`Trainer`] is not responsible for the step 6. The `Agent` does it.
// ///
// /// ## Model evaluation and saving
// ///
// /// [Trainer::train()] evaluates the agent being trained with the interval of optimization
// /// steps specified by [TrainerBuilder::eval_interval()]. If the evaluation reward is
// /// greater than the maximum in the history of training, the agent will be saved in the
// /// directory specified by [TrainerBuilder::model_dir()].
// ///
// /// A trained agent often consists of a number of neural networks like an action-value
// /// network, its target network, a policy network. Typically, [Agent] saves all of these
// /// neural networks in a directory.
// pub struct Trainer<E: Env, A: Agent<E>> {
//     env: E,
//     env_eval: E,
//     agent: A,
//     obs_prev: RefCell<Option<E::Obs>>,
//     max_opts: usize,
//     eval_interval: usize,
//     n_episodes_per_eval: usize,
//     eval_threshold: Option<f32>,
//     model_dir: Option<String>,
//     count_opts: usize,
//     count_steps: usize,
// }

// impl<E: Env, A: Agent<E>> Trainer<E, A> {
//     /// Get the reference to the agent.
//     pub fn get_agent(&self) -> &impl Agent<E> {
//         &self.agent
//     }

//     /// Get the reference to the environment.
//     pub fn get_env(&self) -> &E {
//         &self.env
//     }

//     /// Get the reference to the environment for evaluation.
//     pub fn get_env_eval(&self) -> &E {
//         &self.env_eval
//     }

//     // fn stats_eval_reward(rs: &Vec<f32>) -> (f32, f32, f32) {
//     fn stats_eval_reward(rs: &[f32]) -> (f32, f32, f32) {
//         let mean: f32 = rs.iter().sum::<f32>() / (rs.len() as f32);
//         let min = rs.iter().fold(f32::NAN, |m, v| v.min(m));
//         let max = rs.iter().fold(f32::NAN, |m, v| v.max(m));

//         (mean, min, max)
//     }

//     /// Train the agent.
//     ///
//     /// In the training loop, the following values are recorded in the `recorder`:
//     /// * `n_steps` - The nunber of steps interacting with the environment.
//     /// * `n_opts` - The number of optimization steps.
//     /// * `datetime` - `Date and time`.
//     /// * `mean_cum_eval_reward` - Cumulative rewards in evaluation runs.
//     pub fn train<T: Recorder>(&mut self, recorder: &mut T) {
//         let mut count_steps_local = 0;
//         let mut now = std::time::SystemTime::now();
//         let mut max_eval_reward = std::f32::MIN;

//         let obs = self.env.reset(None).unwrap();
//         self.agent.push_obs(&obs);
//         self.obs_prev.replace(Some(obs));
//         self.agent.train(); // set to training mode

//         loop {
//             let mut over_eval_threshold = false;

//             // For resetted environments, elements in obs_prev are updated with env.reset().
//             // After the update, obs_prev will have o_t+1 without reset, or o_0 with reset.
//             // See `sample()` in `util.rs`.
//             let (step, record_env) = sample(&mut self.env, &mut self.agent, &self.obs_prev);
//             self.count_steps += 1;
//             count_steps_local += 1;

//             // agent.observe() internally creates transisions, i.e., (o_t, a_t, o_t+1, r_t+1).
//             // For o_t, the previous observation stored in the agent is used.
//             let option_record = self.agent.observe(step);

//             // The previous observations in the agent are updated with obs_prev.
//             // These are o_t+1 (without reset) or o_0 (with reset).
//             // In the next iteration of the loop, o_t+1 will be treated as the previous observation
//             // in the next training step executed in agent.observation().
//             self.agent
//                 .push_obs(&self.obs_prev.borrow().as_ref().unwrap());

//             if let Some(mut record) = option_record {
//                 use RecordValue::{DateTime, Scalar};

//                 self.count_opts += 1;
//                 record.insert("n_steps", Scalar(self.count_steps as _));
//                 record.insert("n_opts", Scalar(self.count_opts as _));
//                 record.insert("datetime", DateTime(Local::now()));

//                 if self.count_opts % self.eval_interval == 0 {
//                     // Show FPS before evaluation
//                     let fps = match now.elapsed() {
//                         Ok(elapsed) => {
//                             Some(count_steps_local as f32 / elapsed.as_millis() as f32 * 1000.0)
//                         }
//                         Err(_) => None,
//                     };
//                     // Reset counter for getting FPS in training
//                     count_steps_local = 0;

//                     // The timer is used to measure the elapsed time for evaluation
//                     now = std::time::SystemTime::now();

//                     // Evaluation
//                     self.agent.eval();
//                     let rewards = eval(
//                         &mut self.env_eval,
//                         &mut self.agent,
//                         self.n_episodes_per_eval,
//                     );
//                     let (mean, min, max) = Self::stats_eval_reward(&rewards);
//                     info!(
//                         "Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
//                         self.count_opts, mean, min, max
//                     );
//                     record.insert("mean_cum_eval_reward", Scalar(mean));

//                     if let Some(fps) = fps {
//                         info!("{} FPS in training", fps);
//                     }

//                     match now.elapsed() {
//                         Ok(elapsed) => {
//                             info!("{} sec. in evaluation", elapsed.as_millis() as f32 / 1000.0);
//                         }
//                         Err(_) => {
//                             info!("An error occured when getting time")
//                         }
//                     }

//                     // The timer is used to measure the elapsed time for training
//                     now = std::time::SystemTime::now();

//                     // Save the trained model
//                     if self.model_dir != None && mean > max_eval_reward {
//                         if let Some(model_dir) = self.model_dir.clone() {
//                             max_eval_reward = mean;
//                             match self.agent.save(&model_dir) {
//                                 Ok(()) => info!("Saved the model in {:?}", &model_dir),
//                                 Err(_) => info!("Failed to save model."),
//                             }
//                         }
//                     }

//                     // Set the agent in evaluation mode
//                     self.agent.train();

//                     // If the evaluation reward exceeds the threshold
//                     if let Some(th) = self.eval_threshold {
//                         over_eval_threshold = mean >= th;
//                     }

//                     let record = record.merge(record_env);
//                     recorder.write(record);
//                 }
//             }

//             if self.count_opts >= self.max_opts || over_eval_threshold {
//                 break;
//             }
//         }
//     }
// }
