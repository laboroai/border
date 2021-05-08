//! Trainer.
use anyhow::Result;
use chrono::Local;
use log::info;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[allow(unused_imports)]
use crate::core::{
    record::{RecordValue, Recorder},
    util::{eval, sample},
    Agent, Env, Policy, Step,
};

/// Constructs [Trainer].
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct TrainerBuilder {
    max_opts: usize,
    eval_interval: usize,
    n_episodes_per_eval: usize,
    eval_threshold: Option<f32>,
    model_dir: Option<String>,
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        Self {
            max_opts: 0,
            eval_interval: 0,
            n_episodes_per_eval: 0,
            eval_threshold: None,
            model_dir: None,
        }
    }
}

impl TrainerBuilder {
    /// Set the number of optimization steps.
    pub fn max_opts(mut self, v: usize) -> Self {
        self.max_opts = v;
        self
    }

    /// Set the interval for evaluation.
    pub fn eval_interval(mut self, v: usize) -> Self {
        self.eval_interval = v;
        self
    }

    /// Set the number of episodes for evaluation.
    pub fn n_episodes_per_eval(mut self, v: usize) -> Self {
        self.n_episodes_per_eval = v;
        self
    }

    /// Set the evaluation threshold.
    pub fn eval_threshold(mut self, v: f32) -> Self {
        self.eval_threshold = Some(v);
        self
    }

    /// Set the directory the trained model being saved.
    pub fn model_dir<T: Into<String>>(mut self, model_dir: T) -> Self {
        self.model_dir = Some(model_dir.into());
        self
    }

    /// Constructs [TrainerBuilder] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [TrainerBuilder].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    ///Constructs a trainer.
    pub fn build<E, A>(self, env: E, env_eval: E, agent: A) -> Trainer<E, A>
    where
        E: Env,
        A: Agent<E>,
    {
        Trainer {
            env,
            env_eval,
            agent,
            obs_prev: RefCell::new(None),
            max_opts: self.max_opts,
            eval_interval: self.eval_interval,
            n_episodes_per_eval: self.n_episodes_per_eval,
            eval_threshold: self.eval_threshold,
            model_dir: self.model_dir,
            count_opts: 0,
            count_steps: 0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_serde_trainer_builder() -> Result<()> {
        let trainer = TrainerBuilder::default()
            .max_opts(100)
            .eval_interval(10000)
            .n_episodes_per_eval(5)
            .model_dir("some/directory");

        let dir = TempDir::new("trainer_builder")?;
        let path = dir.path().join("trainer_builder.yaml");
        println!("{:?}", path);

        trainer.save(&path)?;
        let trainer_ = TrainerBuilder::load(&path)?;
        assert_eq!(trainer, trainer_);
        // let yaml = serde_yaml::to_string(&trainer)?;
        // println!("{}", yaml);
        // assert_eq!(
        //     yaml,
        //     "---\n\
        //      max_opts: 100\n\
        //      eval_interval: 10000\n\
        //      n_episodes_per_eval: 5\n\
        //      eval_threshold: ~\n\
        //      model_dir: some/directory\n\
        // "
        // );
        Ok(())
    }
}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Manages training process.
///
/// ## Training loop
///
/// For training an agent with standard RL algorithms in the library, the agent and environment
/// interact as illustrated in the following diagram:
///
/// ```mermaid
/// flowchart TB
///     Trainer -. 0. Env::reset .-> Env
///     Env --> Obs
///     ObsPrev -- 3. Policy::sample --> Policy
///     Policy --> Act
///     Act -- 4. Env::step --> Env
///     Obs --> Step
///     Obs -- 1. RefCell::replace --> ObsPrev
///     Act --> Step
///     ObsPrev -- 2. Agent::push_obs --> ObsPrev'
///     Step -- 5. Agent::observe --> Transition
///
///     subgraph Agent
///         ObsPrev' --> Transition
///         ReplayBuffer -- 6. update policy parameters --- Policy
///         Transition --> ReplayBuffer
///     end
/// ```
///
/// 0. Call [`Env::reset`] for resetting the enbironment and getting an observation.
/// An episode starts.
/// 1. Call [`std::cell::RefCell::replace`] for placing the observation in `PrevObs`.
/// 2. Call [`Agent::push_obs`] for placing the observation in `PrevObs'`.
/// 3. Call [`Policy::sample`] for sampling an action from `Policy`.
/// 4. Call [`Env::step`] for taking an action, getting a new observation, and creating [`Step`] object.
/// 5. Call [`Agent::observe`] for updating the replay buffer with the new and previous observations.
/// 6. Call some methods in the agent for updating policy parameters.
/// 7. Back to 1.
///
/// Actually, [`Trainer`] is not responsible for the step 6. The `Agent` does it.
///
/// ## Model evaluation and saving
///
/// [Trainer::train()] evaluates the agent being trained with the interval of optimization
/// steps specified by [TrainerBuilder::eval_interval()]. If the evaluation reward is
/// greater than the maximum in the history of training, the agent will be saved in the
/// directory specified by [TrainerBuilder::model_dir()].
///
/// A trained agent often consists of a number of neural networks like an action-value
/// network, its target network, a policy network. Typically, [Agent] saves all of these
/// neural networks in a directory.
pub struct Trainer<E: Env, A: Agent<E>> {
    env: E,
    env_eval: E,
    agent: A,
    obs_prev: RefCell<Option<E::Obs>>,
    max_opts: usize,
    eval_interval: usize,
    n_episodes_per_eval: usize,
    eval_threshold: Option<f32>,
    model_dir: Option<String>,
    count_opts: usize,
    count_steps: usize,
}

impl<E: Env, A: Agent<E>> Trainer<E, A> {
    /// Get the reference to the agent.
    pub fn get_agent(&self) -> &impl Agent<E> {
        &self.agent
    }

    /// Get the reference to the environment.
    pub fn get_env(&self) -> &E {
        &self.env
    }

    /// Get the reference to the environment for evaluation.
    pub fn get_env_eval(&self) -> &E {
        &self.env_eval
    }

    // fn stats_eval_reward(rs: &Vec<f32>) -> (f32, f32, f32) {
    fn stats_eval_reward(rs: &[f32]) -> (f32, f32, f32) {
        let mean: f32 = rs.iter().sum::<f32>() / (rs.len() as f32);
        let min = rs.iter().fold(f32::NAN, |m, v| v.min(m));
        let max = rs.iter().fold(f32::NAN, |m, v| v.max(m));

        (mean, min, max)
    }

    /// Train the agent.
    ///
    /// In the training loop, the following values are recorded in the `recorder`:
    /// * `n_steps` - The nunber of steps interacting with the environment.
    /// * `n_opts` - The number of optimization steps.
    /// * `datetime` - `Date and time`.
    /// * `mean_cum_eval_reward` - Cumulative rewards in evaluation runs.
    pub fn train<T: Recorder>(&mut self, recorder: &mut T) {
        let mut count_steps_local = 0;
        let mut now = std::time::SystemTime::now();
        let mut max_eval_reward = std::f32::MIN;

        let obs = self.env.reset(None).unwrap();
        self.agent.push_obs(&obs);
        self.obs_prev.replace(Some(obs));
        self.agent.train(); // set to training mode

        loop {
            let mut over_eval_threshold = false;

            // For resetted environments, elements in obs_prev are updated with env.reset().
            // After the update, obs_prev will have o_t+1 without reset, or o_0 with reset.
            // See `sample()` in `util.rs`.
            let (step, _) = sample(&mut self.env, &mut self.agent, &self.obs_prev);
            self.count_steps += 1;
            count_steps_local += 1;

            // agent.observe() internally creates transisions, i.e., (o_t, a_t, o_t+1, r_t+1).
            // For o_t, the previous observation stored in the agent is used.
            let option_record = self.agent.observe(step);

            // The previous observations in the agent are updated with obs_prev.
            // These are o_t+1 (without reset) or o_0 (with reset).
            // In the next iteration of the loop, o_t+1 will be treated as the previous observation
            // in the next training step executed in agent.observation().
            self.agent
                .push_obs(&self.obs_prev.borrow().as_ref().unwrap());

            if let Some(mut record) = option_record {
                use RecordValue::{DateTime, Scalar};

                self.count_opts += 1;
                record.insert("n_steps", Scalar(self.count_steps as _));
                record.insert("n_opts", Scalar(self.count_opts as _));
                record.insert("datetime", DateTime(Local::now()));

                if self.count_opts % self.eval_interval == 0 {
                    // Show FPS before evaluation
                    let fps = match now.elapsed() {
                        Ok(elapsed) => {
                            Some(count_steps_local as f32 / elapsed.as_millis() as f32 * 1000.0)
                        }
                        Err(_) => None,
                    };
                    // Reset counter for getting FPS in training
                    count_steps_local = 0;

                    // The timer is used to measure the elapsed time for evaluation
                    now = std::time::SystemTime::now();

                    // Evaluation
                    self.agent.eval();
                    let rewards = eval(
                        &mut self.env_eval,
                        &mut self.agent,
                        self.n_episodes_per_eval,
                    );
                    let (mean, min, max) = Self::stats_eval_reward(&rewards);
                    info!(
                        "Opt step {}, Eval (mean, min, max) of r_sum: {}, {}, {}",
                        self.count_opts, mean, min, max
                    );
                    record.insert("mean_cum_eval_reward", Scalar(mean));

                    if let Some(fps) = fps {
                        info!("{} FPS in training", fps);
                    }

                    match now.elapsed() {
                        Ok(elapsed) => {
                            info!("{} sec. in evaluation", elapsed.as_millis() as f32 / 1000.0);
                        }
                        Err(_) => {
                            info!("An error occured when getting time")
                        }
                    }

                    // The timer is used to measure the elapsed time for training
                    now = std::time::SystemTime::now();

                    // Save the trained model
                    if self.model_dir != None && mean > max_eval_reward {
                        if let Some(model_dir) = self.model_dir.clone() {
                            max_eval_reward = mean;
                            match self.agent.save(&model_dir) {
                                Ok(()) => info!("Saved the model in {:?}", &model_dir),
                                Err(_) => info!("Failed to save model."),
                            }
                        }
                    }

                    // Set the agent in evaluation mode
                    self.agent.train();

                    // If the evaluation reward exceeds the threshold
                    if let Some(th) = self.eval_threshold {
                        over_eval_threshold = mean >= th;
                    }

                    recorder.write(record);
                }
            }

            if self.count_opts >= self.max_opts || over_eval_threshold {
                break;
            }
        }
    }
}
