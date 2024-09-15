//! Interfaces for Minari datasets.
//!
//! ```no_run
//! fn main() -> Result<()> {
//!     let dataset = MinariDataset::load_dataset("D4RL/kitchen/complete-v1", true)?;
//!
//!     // The number of transitions over all episodes
//!     let num_transitions = dataset.get_num_transitions(None)?;
//!     println!("{:?}", num_transitions);
//!
//!     // Converter for observation and action
//!     let converter = KitchenNdarrayConverter {};
//!
//!     // Create replay buffer for the sixth episode
//!     let replay_buffer = dataset.create_replay_buffer(&converter, Some(vec![5]))?;
//!
//!     // Recover the environment from the dataset
//!     let mut env = dataset.recover_environment(converter, false, "human")?;
//!
//!     // Sequence of actions in the episode
//!     let actions = replay_buffer.whole_actions();
//!
//!     // Apply the actions to the environment
//!     env.reset(None)?;
//!     for ix in 0..actions.action.shape()[0] {
//!         let act = actions.get(ix);
//!         let _ = env.step(&act);
//!     }
//!
//!     Ok(())
//! }
//! ```
use anyhow::Result;
use border_core::{
    generic_replay_buffer::{
        BatchBase, GenericTransitionBatch, SimpleReplayBuffer, SimpleReplayBufferConfig,
    },
    record::{Record, RecordValue::Scalar},
    Act, Env, ExperienceBufferBase, Obs, ReplayBufferBase, Step,
};
use pyo3::{
    types::{IntoPyDict, PyIterator, PyTuple},
    FromPyObject, PyAny, PyObject, Python, ToPyObject,
};
use std::marker::PhantomData;
pub mod d4rl;
pub mod util;

/// Conversion trait for observation and action.
pub trait MinariConverter {
    /// Observation type.
    type Obs: Obs;

    /// Action type.
    type Act: Act;

    /// Batch of observation.
    type ObsBatch: BatchBase + From<Self::Obs>;

    /// Batch of action.
    type ActBatch: BatchBase + From<Self::Act>;

    /// Converts [`PyObject`] into [`Self::Obs`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_observation(&self, obj: &PyAny) -> Result<Self::Obs>;

    /// Converts [`Self::Act`] into [`PyObject`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_action(&self, act: Self::Act) -> Result<PyObject>;

    /// Converts [`PyObject`] of an episode into [`Self::ObsBatch`].
    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch>;

    /// Converts [`PyObject`] of an episode into [`Self::ObsBatch`] for `next_obs`.
    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch>;

    /// Converts [`PyObject`] of an episode into [`Self::ActBatch`].
    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch>;
}

/// Common interface for Minari datasets.
pub struct MinariDataset {
    dataset: PyObject,
}

impl MinariDataset {
    /// Loads dataset.
    ///
    /// * `dataset_id`: name id of Minari dataset.
    /// * `download`: if `true`, download the dataset. if it is not found locally.
    pub fn load_dataset(dataset_id: impl AsRef<str>, download: bool) -> Result<Self> {
        Python::with_gil(|py| {
            let minari = py.import("minari").unwrap();
            let dataset = minari
                .call1("load_dataset", (dataset_id.as_ref(), download))?
                .to_object(py);
            Ok(Self { dataset })
        })
    }

    /// Gets the number of transitions over all episodes.
    pub fn get_num_transitions(&self, episode_indices: Option<Vec<usize>>) -> Result<usize> {
        Python::with_gil(|py| {
            let episodes = self
                .dataset
                .call_method1(py, "iterate_episodes", (episode_indices,))?;
            let mut total = 0;

            // Iterate over episodes
            for ep in PyIterator::from_object(py, &episodes)? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;
                total += ep
                    .getattr("rewards")?
                    .call_method0("__len__")?
                    .extract::<usize>()?;
            }
            Ok(total)
        })
    }

    /// Creates replay buffer from the dataset.
    ///
    /// The order of transitions in the original dataset is preserved,
    /// but the boundary between episodes is discarded.
    ///
    /// * `converter`: converter for observation and action.
    /// * `episode_indices`: indices of episodes to be included in the replay buffer.
    ///   If `None`, all episodes are included.
    ///   If `Some(indices)`, only episodes with the indices are included.
    ///   All transitions in the episodes are flattened.
    pub fn create_replay_buffer<T: MinariConverter>(
        &self,
        converter: &T,
        episode_indices: Option<Vec<usize>>,
    ) -> Result<SimpleReplayBuffer<T::ObsBatch, T::ActBatch>>
    where
        T::ObsBatch: std::fmt::Debug,
        T::ActBatch: std::fmt::Debug,
    {
        Python::with_gil(|py| {
            // The total number of transitions
            let num_transitions = self.get_num_transitions(episode_indices.clone())?;

            // Prepare replay buffer
            let mut replay_buffer = SimpleReplayBuffer::build(&SimpleReplayBufferConfig {
                capacity: num_transitions,
                seed: 0,
                per_config: None,
            });

            let episodes = self
                .dataset
                .call_method1(py, "iterate_episodes", (episode_indices,))?;

            // Iterate over episodes
            for ep in PyIterator::from_object(py, &episodes)? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;

                // Extract transitions in the episode as a batch
                let batch = Self::extract_transitions_in_episode(py, &ep, converter)?;

                // Push the batch to the replay buffer
                replay_buffer.push(batch)?;
            }

            Ok(replay_buffer)
        })
    }

    fn extract_transitions_in_episode<T: MinariConverter>(
        py: Python,
        ep: &PyAny,
        converter: &T,
    ) -> Result<GenericTransitionBatch<T::ObsBatch, T::ActBatch>>
    where
        T::ObsBatch: std::fmt::Debug,
        T::ActBatch: std::fmt::Debug,
    {
        // Extract episode as batch
        let obs = ep.getattr("observations")?;
        let act = ep.getattr("actions")?;
        let rew = ep.getattr("rewards")?;
        let trm = ep.getattr("terminations")?;
        let trn = ep.getattr("truncations")?;

        // Creates batch
        let next_obs = converter.convert_observation_batch_next(&obs)?;
        let obs = converter.convert_observation_batch(&obs)?;
        let act = converter.convert_action_batch(&act)?;
        let reward = util::vec::pyany_to_f32vec(py, &rew)?;
        let is_terminated = util::vec::pyany_to_vec::<i8>(py, &trm)?;
        let is_truncated = util::vec::pyany_to_vec::<i8>(py, &trn)?;

        Ok(GenericTransitionBatch {
            obs,
            act,
            next_obs,
            reward,
            is_terminated,
            is_truncated,
            weight: None,
            ix_sample: None,
        })
    }

    /// Recovers the environment.
    ///
    /// * `converter`: converter for observation and action.
    /// * `eval_env`: if `true`, the environment is for evaluation.
    ///   See [Minari API documentation](https://minari.farama.org/api/minari_dataset/minari_dataset/#minari.MinariDataset.recover_environment).
    /// * `render_mode`: render mode for the environment.
    pub fn recover_environment<'a, T, O, A>(
        &self,
        converter: T,
        eval_env: bool,
        render_mode: impl Into<Option<&'a str>>,
    ) -> Result<MinariEnv<T, O, A>>
    where
        T: MinariConverter<Obs = O, Act = A>,
        O: Obs,
        A: Act,
    {
        let env = {
            Python::with_gil(|py| {
                let kwargs = [("render_mode", render_mode.into())].into_py_dict(py);
                let env =
                    self.dataset
                        .call_method(py, "recover_environment", (eval_env,), Some(&kwargs));
                env
                // env.to_object(py)
            })?
        };

        Ok(MinariEnv {
            converter,
            env,
            initial_seed: None,
            count_steps: 0,
            max_steps: None,
            phantom: PhantomData,
        })
    }
}

/// Environment interface for Minari datasets.
pub struct MinariEnv<T, O, A>
where
    T: MinariConverter<Obs = O, Act = A>,
    O: Obs,
    A: Act,
{
    converter: T,
    env: PyObject,
    initial_seed: Option<i64>,
    count_steps: usize,
    max_steps: Option<usize>,
    phantom: PhantomData<(O, A)>,
}

impl<T, O, A> Env for MinariEnv<T, O, A>
where
    T: MinariConverter<Obs = O, Act = A>,
    O: Obs,
    A: Act,
{
    type Obs = O;
    type Act = A;
    type Config = ();
    type Info = ();

    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
        // Reset the environment
        let reset = match is_done {
            None => true,
            // when reset() is called in border_core::util::sample()
            Some(v) => {
                debug_assert_eq!(v.len(), 1);
                v[0] != 0
            }
        };

        if !reset {
            Ok(O::dummy(1))
        } else {
            pyo3::Python::with_gil(|py| {
                let ret_values = if let Some(seed) = self.initial_seed {
                    self.initial_seed = None;
                    let kwargs = Some(vec![("seed", seed)].into_py_dict(py));
                    self.env.call_method(py, "reset", (), kwargs)?
                } else {
                    self.env.call_method0(py, "reset")?
                };
                let ret_values_: &PyTuple = ret_values.extract(py)?;
                self.converter
                    .convert_observation(ret_values_.get_item(0).extract()?)
            })
        }
    }

    fn step(&mut self, act: &Self::Act) -> (Step<Self>, Record) {
        fn is_done(step: &PyTuple) -> (i8, i8) {
            // terminated or truncated
            let is_terminated = match step.get_item(2).extract().unwrap() {
                true => 1,
                false => 0,
            };
            let is_truncated = match step.get_item(3).extract().unwrap() {
                true => 1,
                false => 0,
            };

            (is_terminated, is_truncated)
        }

        pyo3::Python::with_gil(|py| {
            // if self.render {
            //     if !self.pybullet {
            //         let _ = self.env.call_method0(py, "render");
            //     } else {
            //         let cam: &PyModule = self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
            //         // cam.call1("update_camera_pos", (&self.env,)).unwrap();
            //         cam.getattr("update_camera_pos")
            //             .unwrap()
            //             .call1((&self.env,))
            //             .unwrap();
            //     }
            //     std::thread::sleep(self.wait);
            // }

            // State transition
            let (
                act,
                next_obs,
                reward,
                is_terminated,
                mut is_truncated,
                mut record,
                info,
                init_obs,
            ) = {
                let a_py = self.converter.convert_action(act.clone()).unwrap();
                let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
                let step: &PyTuple = ret.extract(py).unwrap();
                let next_obs = self
                    .converter
                    .convert_observation(step.get_item(0))
                    .unwrap();
                let reward: Vec<f32> = vec![step.get_item(1).extract().unwrap()];
                let (is_terminated, is_truncated) = is_done(step);
                let is_terminated = vec![is_terminated];
                let is_truncated = vec![is_truncated];
                let info = ();
                let init_obs = O::dummy(1);
                let act = act.clone();

                (
                    act,
                    next_obs,
                    reward,
                    is_terminated,
                    is_truncated,
                    Record::empty(),
                    info,
                    init_obs,
                )
            };

            self.count_steps += 1; //.replace(c + 1);

            // Terminated or truncated
            if let Some(max_steps) = self.max_steps {
                if self.count_steps >= max_steps {
                    is_truncated[0] = 1;
                }
            };

            if (is_terminated[0] | is_truncated[0]) == 1 {
                record.insert("episode_length", Scalar(self.count_steps as _));
                self.count_steps = 0;
            }

            (
                Step::new(
                    next_obs,
                    act,
                    reward,
                    is_terminated,
                    is_truncated,
                    info,
                    init_obs,
                ),
                record,
            )
        })
    }

    fn build(config: &Self::Config, seed: i64) -> Result<Self>
    where
        Self: Sized,
    {
        unimplemented!("Use MinariDataset::recover_environment()");
    }

    fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs> {
        unimplemented!();
    }
}

impl<T, O, A> Drop for MinariEnv<T, O, A>
where
    T: MinariConverter<Obs = O, Act = A>,
    O: Obs,
    A: Act,
{
    fn drop(&mut self) {
        // Call closd()
        // See https://gymnasium.farama.org/api/env/#gymnasium.Env.close
        Python::with_gil(|py| {
            self.env.call_method0(py, "close").unwrap();
        })
    }
}
