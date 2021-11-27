use anyhow::Result;
use border_core::{
    record::{Record, RecordValue::Scalar, Recorder},
    Agent, Env, Obs, ReplayBufferBase, StepProcessorBase,
};
use crossbeam_channel::unbounded;
// pub use config::TrainerConfig;
use log::info;
// pub use sampler::SyncSampler;

/// Manages asynchronous training loop in a single machine.
///
/// It will be used for off-policy algorithms.
pub struct AsyncTrainer {
    /// Where to save the trained model.
    pub model_dir: Option<String>,

    /// Interval of optimization in environment steps.
    pub opt_interval: usize,

    /// Interval of recording in training steps.
    pub record_interval: usize,

    /// Interval of evaluation in training steps.
    pub eval_interval: usize,

    /// The maximal number of training steps.
    pub max_train_steps: usize,

    /// Interval of synchronizing model parameters in training steps.
    pub sync_interval: usize,
}

impl AsyncTrainer {
    fn do_eval(&self, agent: &mut A, record: &mut Record, max_eval_reward: f32) -> Result<()> {
        let eval_reward = self.evaluate(agent)?;
        record.insert("eval_reward", Scalar(eval_reward));

        // Save the best model up to the current iteration
        if eval_reward > max_eval_reward {
            let model_dir = self.model_dir.as_ref().unwrap().clone();
            let model_dir = model_dir + "/best";
            match agent.save(&model_dir) {
                Ok(()) => info!("Saved the model in {:?}", &model_dir),
                Err(_) => info!("Failed to save model."),
            };
            eval_reward
        } else {
            max_eval_reward
        }
    }

    fn save_model<A: Agent<E, R>>(&self, agent: &A, train_steps: usize) {
        let model_dir = self.model_dir.as_ref().unwrap().clone();
        let model_dir = model_dir + format!("/{}", train_steps).as_str();

        match agent.save(&model_dir) {
            Ok(()) => info!("Saved the model in {:?}", &model_dir),
            Err(_) => info!("Failed to save model."),
        }
    }

    pub fn train(&mut self, agent: &mut A, recorder: &mut S)
    where
        A: Agent<E, R>,
        S: Recorder,
    {
        let mut max_eval_reward = -f32::MAX;
        let mut train_steps_: usize = 0; // computing training steps per second
        let mut train_time_: f32 = 0.;

        // let (s1, r3) = run_replay_buffer();
        // let s2 = run_actors(s1);
        // s1: send messages from the actors and the trainer to the replay buffer
        // s2: send messages from the trainer to the actors (samplers)
        // r3: receive messages from the replay buffer

        // wait until taking enough samples before the start of training
        self.wait(&s1, &r3);

        for train_steps in 1..=self.max_train_steps {
            let (record, time) = self.train_step(agent, &s1, &r3);

            train_steps_ += 1;
            train_time_ += time.as_millis() as f32;

            // Synchronization, evaluation, recording, saving
            let do_eval = train_steps % self.eval_interval == 0;
            let do_rec = train_steps % self.record_interval == 0;
            let do_sync = train_steps % self.sync_interval == 0;
            let do_save = train_steps % self.save_interval == 0;

            if do_sync {
                self.sync_model_params();
            }

            if do_eval {
                max_eval_reward = self.do_eval(agent, &mut record, max_eval_reward);
            }

            if do_rec {
                self.do_record(&mut record, train_steps_, train_time_);
                train_steps_ = 0;
                train_time_ = 0.;
            }

            if do_save {
                self.save_model(agent, train_steps);
            }

            // Flush record to the recorder
            if do_eval || do_rec {
                recorder.write(record);
            }
        }

        //
    }
}

// pub struct AsyncTrainer<E, P, R>
// where
//     E: Env,
//     P: StepProcessorBase<E>,
//     R: ReplayBufferBase<PushedItem = P::Output>,
// {

// }

// impl<E, P, R> AsyncTrainer<E, P, R>
// where
//     E: Env,
//     P: StepProcessorBase<E>,
//     R: ReplayBufferBase<PushedItem = P::Output>,
// {
//     /// Trains the agent asynchronously.
//     pub fn train<A, S>(&mut self, agent: &mut A, recorder: &mut S) -> Result<()>
//     where
//         A: Agent<E, R>,
//         S: Recorder,
//     {
//         // Channels for sending samples from actors to the buffer
//         let (s_samples, r_samples) = unbounded();
//         // Channels for sending model parameters from the learner to actors
//         let (s_params, r_params) = unbounded();
//         // Channels for

//         // creates and launches actors (samplers)
//         self.launch_samplers(s_samples, r_params);
//     }
// }
