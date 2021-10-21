//! Samples transitions and pushes them into a replay buffer.
use crate::{Env, Agent, ReplayBufferBase, StepProcessorBase, record::Record};
use anyhow::Result;

pub struct SyncSampler<E, P>
where
    E: Env,
    P: StepProcessorBase<E>,
{
    env: E,
    prev_obs: Option<E::Obs>,
    producer: P,
    n_frames: usize,
    time: f32,
}

impl<E, P> SyncSampler<E, P>
where
    E: Env,
    P: StepProcessorBase<E>,
{
    /// Creates a sampler.
    pub fn new(env: E, producer: P) -> Self {
        Self {
            env,
            prev_obs: None,
            producer,
            n_frames: 0,
            time: 0f32,
        }
    }

    /// Samples transitions and pushes them into the replay buffer.
    pub fn sample_and_push<A, R>(&mut self, agent: &mut A, buffer: &mut R) -> Result<Record> 
    where
        A: Agent<E, R>,
        R: ReplayBufferBase<PushedItem = P::Output>
    {
        let now = std::time::SystemTime::now();

        // Reset environment(s) if required
        if self.prev_obs.is_none() {
            // For a vectorized environments, reset all environments in `env`
            // by giving `None` to reset() method
            self.prev_obs = Some(self.env.reset(None)?);
            self.producer.reset(self.prev_obs.as_ref().unwrap().clone());
        }

        // Sample action(s) and apply it to environment(s)
        let act = agent.sample(self.prev_obs.as_ref().unwrap());
        let (step, record) = self.env.step_with_reset(&act);
        self.prev_obs = Some(step.obs.clone());

        // Create and push transition(s)
        let transition = self.producer.process(step);
        buffer.push(transition);

        // For counting FPS
        if let Ok(time) = now.elapsed() {
            self.n_frames += 1;
            self.time += time.as_millis() as f32;
        }

        Ok(record)
    }

    /// Returns FPS, including taking action, applying it to the, environment,
    /// producing transition, and pushing it into the replay buffer.
    pub fn fps(&self) -> f32 {
        self.n_frames as f32 / self.time * 1000f32
    }

    /// Reset stats for computing FPS.
    pub fn reset(&mut self) {
        self.n_frames = 0;
        self.time = 0f32;
    }
}
