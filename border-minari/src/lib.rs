//! Interface to access [Minari](https://minari.farama.org/index.html#) datasets.
//!
//! [`MinariDataset`] and [`MinariEnv`] provide a common interface to access Minari datasets.
//! These structs are used with concrete observation and action types.
//! For example, [`border_minari::d4rl::kitchen`] provides observation and action types,
//! and the corresponding converter for the [Kitchen datasets](https://minari.farama.org/datasets/D4RL/kitchen/).
//!
//! The implementation of data types and converters depends on the backend for implementing
//! your agents. In [`border_minari::d4rl::kitchen::ndarray`], the observation and action types are
//! defined essentially as [`ndarray::ArrayD`]. In [`border_minari::d4rl::kitchen::candle`],
//! the observation and action types are defined as [`candle_core::Tensor`].
//!
//! In the below example, we load an episode in the Kitchen dataset and create a replay buffer for that.
//! Then, we recover the environment from the dataset and apply the actions in the episode.
//! The observation and action types are implemented with [`ndarray::ArrayD`].
//!
//! ```no_run
//! # use anyhow::Result;
//! use border_core::Env;
//! use border_minari::{d4rl::kitchen::ndarray::KitchenConverter, MinariDataset};
//! # use numpy::convert;
//! # use std::num;
//!
//! fn main() -> Result<()> {
//!     let dataset = MinariDataset::load_dataset("D4RL/kitchen/complete-v1", true)?;
//!
//!     // Converter for observation and action
//!     let converter = KitchenConverter {};
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
//! [`candle_core::Tensor`]: candle_core::Tensor
//! [`border_minari::d4rl::kitchen`]: crate::d4rl::kitchen
//! [`border_minari::d4rl::kitchen::ndarray`]: crate::d4rl::kitchen::ndarray
//! [`border_minari::d4rl::kitchen::candle`]: crate::d4rl::kitchen::candle
use anyhow::Result;
use border_core::{
    generic_replay_buffer::{
        BatchBase, GenericTransitionBatch, SimpleReplayBuffer, SimpleReplayBufferConfig,
    },
    record::{Record, RecordValue::Scalar},
    Act, Env, ExperienceBufferBase, Obs, ReplayBufferBase, Step,
};
use pyo3::{
    py_run,
    types::{IntoPyDict, PyDict, PyIterator, PyTuple},
    PyAny, PyObject, Python, ToPyObject,
};
pub mod d4rl;
mod evaluator;
pub mod util;
pub use evaluator::MinariEvaluator;

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
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_observation_batch(&self, obj: &PyAny) -> Result<Self::ObsBatch>;

    /// Converts [`PyObject`] of an episode into [`Self::ObsBatch`] for `next_obs`.
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_observation_batch_next(&self, obj: &PyAny) -> Result<Self::ObsBatch>;

    /// Converts [`PyObject`] of an episode into [`Self::ActBatch`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_action_batch(&self, obj: &PyAny) -> Result<Self::ActBatch>;

    /// Returns optional parameters when recovering the environment for evaluation from the dataset.
    fn env_params(&self, py: Python<'_>) -> Vec<(&str, PyObject)>;
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
                .getattr("load_dataset")?
                .call1((dataset_id.as_ref(), download))?
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
        converter: &mut T,
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

            // Stats in the replay buffer
            log::info!("In replay buffer:");
            log::info!("{} transitions", num_transitions);
            log::info!("{} terminated flags", replay_buffer.num_terminated_flags());
            log::info!("{} truncated flags", replay_buffer.num_truncated_flags());
            log::info!("{} reward sum", replay_buffer.sum_rewards());

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
    pub fn recover_environment<'a, T: MinariConverter>(
        &self,
        converter: T,
        eval_env: bool,
        render_mode: impl Into<Option<&'a str>>,
    ) -> Result<MinariEnv<T>> {
        let env = {
            Python::with_gil(|py| {
                let mut kwargs: Vec<(&str, PyObject)> =
                    vec![("render_mode", render_mode.into().to_object(py))];
                kwargs.extend(converter.env_params(py));
                let kwargs = kwargs.into_py_dict(py);
                let env =
                    self.dataset
                        .call_method(py, "recover_environment", (eval_env,), Some(&kwargs));
                env
            })?
        };

        let ref_score_minmax = Python::with_gil(|py| {
            let min = self
                .dataset
                .getattr(py, "storage")
                .unwrap()
                .getattr(py, "metadata")
                .unwrap()
                .call_method1(py, "get", ("ref_min_score",));
            let max = self
                .dataset
                .getattr(py, "storage")
                .unwrap()
                .getattr(py, "metadata")
                .unwrap()
                .call_method1(py, "get", ("ref_max_score",));

            if min.is_err() || max.is_err() {
                log::info!("Either ref_min_score or ref_max_score is missing. Normalized score cannot be calculated.");
                None
            } else {
                let min = min.unwrap().extract(py);
                let max = max.unwrap().extract(py);
                if min.is_ok() && max.is_ok() {
                    let min = min.unwrap();
                    let max = max.unwrap();
                    log::info!("ref_min_score = {min}");
                    log::info!("ref_max_score = {max}");
                    Some((min, max))
                } else {
                    log::info!("Either ref_min_score or ref_max_score could not be converted to f32. Normalized score cannot be calculated.");
                    None
                }
            }
        });

        Ok(MinariEnv {
            converter,
            env,
            initial_seed: None,
            count_steps: 0,
            max_steps: None,
            ref_score_minmax,
            // dataset: self.dataset.clone(),
        })
    }
}

/// Environment interface for Minari datasets.
pub struct MinariEnv<T: MinariConverter> {
    converter: T,
    env: PyObject,
    initial_seed: Option<i64>,
    count_steps: usize,
    max_steps: Option<usize>,
    ref_score_minmax: Option<(f32, f32)>,
}

impl<T: MinariConverter> Env for MinariEnv<T> {
    type Obs = T::Obs;
    type Act = T::Act;
    type Config = ();
    type Info = ();

    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
        assert_eq!(is_done, None);

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
                let init_obs = None;
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

    fn build(_config: &Self::Config, _seed: i64) -> Result<Self>
    where
        Self: Sized,
    {
        unimplemented!("Use MinariDataset::recover_environment()");
    }

    /// For this environment, `ix` is ignored for now.
    fn reset_with_index(&mut self, _ix: usize) -> Result<Self::Obs> {
        self.reset(None)
    }
}

impl<T: MinariConverter> Drop for MinariEnv<T> {
    fn drop(&mut self) {
        // Call closd()
        // See https://gymnasium.farama.org/api/env/#gymnasium.Env.close
        Python::with_gil(|py| {
            self.env.call_method0(py, "close").unwrap();
        })
    }
}

impl<T: MinariConverter> MinariEnv<T> {
    /// Normalize undiscounted return of an episode.
    ///
    /// This method internally calls [minari.get_normalized_score()](https://minari.farama.org/api/minari_functions/#normalize-score).
    pub fn get_normalized_score(&self, raw_score: f32) -> Option<f32> {
        if let Some((min, max)) = self.ref_score_minmax {
            Some((raw_score - min) / (max - min))
        } else {
            None
        }
    }

    pub fn get_env_params(&self) -> Result<Vec<(String, String)>> {
        let mut result = vec![];
        pyo3::Python::with_gil(|py| {
            let builtins = pyo3::types::PyModule::import(py, "builtins")?;
            let print = builtins.getattr("print")?;

            // let dict1 = self.env.getattr(py, "__dict__")?;
            // let keys = dict1.call_method0(py, "keys")?;
            // py_run!(py, keys, "
            //     for key in keys:
            //         print(key)
            // ");
            let env = &self.env;
            py_run!(py, env, "print(env)");
            py_run!(py, env, "print(env._max_episode_steps)");
            let dict = self
                .env
                .getattr(py, "env")?
                .getattr(py, "env")?
                .getattr(py, "env")?
                .getattr(py, "__dict__")?;
            py_run!(py, dict, "print('reward_type     = ', dict['reward_type'])");
            py_run!(
                py,
                dict,
                "print('continuing_task = ', dict['continuing_task'])"
            );
            py_run!(
                py,
                dict,
                "print('reset_target    = ', dict['reset_target'])"
            );
            py_run!(py, dict, "print('spec            = ', dict['spec'])");

            // panic!();
            Ok(result)
        })
    }
}
