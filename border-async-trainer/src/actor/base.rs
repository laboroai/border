use crate::{ActorStat, PushedItemMessage, ReplayBufferProxy, ReplayBufferProxyConfig, SyncModel};
use border_core::{
    Agent, Configurable, Env, ExperienceBufferBase, ReplayBufferBase, Sampler, StepProcessor,
};
use crossbeam_channel::Sender;
use log::info;
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Generate transitions by running [`Agent`] in [`Env`].
///
/// ```mermaid
/// flowchart TB
///   E["Agent<br>(in AsyncTrainer)"]-->|SyncModel::ModelInfo|A
///   subgraph D[Actor]
///     A[Agent]-->|Env::Act|B[Env]
///     B-->|Env::Obs|A
///     B-->|Step&ltE: Env&gt|C[StepProcessor]
///   end
///   C-->|ReplayBufferBase::PushedItem|F[ReplayBufferProxy]
/// ```
///
/// In [`Actor`], an [`Agent`] runs on an [`Env`] and generates [`Step`] objects. 
/// These objects are processed with [`StepProcessor`] and sent to [`ReplayBufferProxy`].
/// The [`Agent`] in the [`Actor`] periodically synchronizes with the [`Agent`] in
/// [`AsyncTrainer`] via [`SyncModel::ModelInfo`].
///
/// See also the diagram in [`AsyncTrainer`].
///
/// [`AsyncTrainer`]: crate::AsyncTrainer
/// [`Agent`]: border_core::Agent
/// [`Env`]: border_core::Env
/// [`StepProcessor`]: border_core::StepProcessor
/// [`Step`]: border_core::Step
pub struct Actor<A, E, P, R>
where
    A: Agent<E, R> + Configurable<E> + SyncModel,
    E: Env,
    P: StepProcessor<E>,
    R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
{
    /// Stops sampling process if this field is set to `true`.
    id: usize,
    // #[allow(dead_code)] // TODO: remove this
    stop: Arc<Mutex<bool>>,
    agent_config: A::Config,
    env_config: E::Config,
    step_proc_config: P::Config,
    replay_buffer_config: ReplayBufferProxyConfig,
    #[allow(dead_code)] // TODO: remove this
    env_seed: i64,
    stats: Arc<Mutex<Option<ActorStat>>>,
    phantom: PhantomData<(A, E, P, R)>,
}

impl<A, E, P, R> Actor<A, E, P, R>
where
    A: Agent<E, R> + Configurable<E> + SyncModel,
    E: Env,
    P: StepProcessor<E>,
    R: ExperienceBufferBase<Item = P::Output> + ReplayBufferBase,
{
    pub fn build(
        id: usize,
        agent_config: A::Config,
        env_config: E::Config,
        step_proc_config: P::Config,
        replay_buffer_config: ReplayBufferProxyConfig,
        stop: Arc<Mutex<bool>>,
        env_seed: i64,
        stats: Arc<Mutex<Option<ActorStat>>>,
    ) -> Self {
        log::info!("Create actor {}", id);
        Self {
            id,
            stop,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            step_proc_config: step_proc_config.clone(),
            replay_buffer_config: replay_buffer_config.clone(),
            env_seed,
            stats,
            phantom: PhantomData,
        }
    }

    fn sync_model_first(agent: &mut A, model_info: &Arc<Mutex<(usize, A::ModelInfo)>>, id: usize) {
        let model_info = model_info.lock().unwrap();
        agent.sync_model(&model_info.1);
        info!("Received the initial model info in actor {}", id);
    }

    fn sync_model(
        agent: &mut A,
        n_opt_steps: &mut usize,
        model_info: &Arc<Mutex<(usize, A::ModelInfo)>>,
        id: usize,
    ) {
        let model_info = model_info.lock().unwrap();
        if model_info.0 > *n_opt_steps {
            *n_opt_steps = model_info.0;
            agent.sync_model(&model_info.1);
            info!(
                "Synchronized the model info of {} opt steps in actor {}",
                n_opt_steps, id
            );
        }
    }

    /// Runs sampling loop until `self.stop` becomes `true`.
    ///
    /// When finishes, this method sets [ActorStat].
    pub fn run(
        &mut self,
        sender: Sender<PushedItemMessage<R::Item>>,
        model_info: Arc<Mutex<(usize, A::ModelInfo)>>,
        guard: Arc<Mutex<bool>>,
        guard_init_model: Arc<Mutex<bool>>,
    ) {
        let mut agent = A::build(self.agent_config.clone());
        let mut buffer =
            ReplayBufferProxy::<R>::build_with_sender(self.id, &self.replay_buffer_config, sender);
        let mut sampler = {
            let mut tmp = guard.lock().unwrap();
            let env = E::build(&self.env_config, self.env_seed).unwrap();
            let step_proc = P::build(&self.step_proc_config);
            *tmp = true;
            Sampler::new(env, step_proc)
        };
        info!("Starts actor {:?}", self.id);

        let mut env_steps = 0;
        let mut n_opt_steps = 0;
        let time = std::time::SystemTime::now();

        // Waits and syncs the initial model
        {
            let mut guard_init_model = guard_init_model.lock().unwrap();
            Self::sync_model_first(&mut agent, &model_info, self.id);
            *guard_init_model = true;
        }

        // Set agent training mode for exploration
        agent.train();

        // Sampling loop
        info!("Starts sampling loop in actor {}", self.id);
        loop {
            // Check model update and synchronize
            Self::sync_model(&mut agent, &mut n_opt_steps, &model_info, self.id);

            // TODO: error handling
            let _record = sampler.sample_and_push(&mut agent, &mut buffer).unwrap();
            env_steps += 1;

            // Stop sampling loop
            if *self.stop.lock().unwrap() {
                *self.stats.lock().unwrap() = Some(ActorStat {
                    env_steps,
                    duration: time.elapsed().unwrap(),
                });
                break;
            }
        }
        info!("Stopped thread for actor {}", self.id);
    }
}
