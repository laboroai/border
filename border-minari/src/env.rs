//! Environment interface for Minari datasets.
//!
//! This module provides a Rust implementation of the environment interface for Minari datasets.
//! It allows for the interaction with environments defined in Minari datasets using the Border framework.
//!
//! The `MinariEnv` struct is the main entry point for interacting with Minari environments.
//! It implements the `Env` trait from the `border-core` crate, which provides a common interface for interacting with environments.
use crate::MinariConverter;
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue::Scalar},
    Env, Step,
};
use pyo3::{
    types::{IntoPyDict, PyTuple},
    PyObject, Python,
};

/// Environment interface for Minari datasets.
pub struct MinariEnv<T: MinariConverter> {
    pub(crate) converter: T,
    pub(crate) env: PyObject,
    pub(crate) initial_seed: Option<i64>,
    pub(crate) count_steps: usize,
    pub(crate) max_steps: Option<usize>,
    pub(crate) ref_score_minmax: Option<(f32, f32)>,
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

    // For debug
    // pub fn get_env_params(&self) -> Result<Vec<(String, String)>> {
    //     let result = vec![];
    //     pyo3::Python::with_gil(|py| {
    //         let builtins = pyo3::types::PyModule::import(py, "builtins")?;
    //         let _print = builtins.getattr("print")?;

    //         // let dict1 = self.env.getattr(py, "__dict__")?;
    //         // let keys = dict1.call_method0(py, "keys")?;
    //         // py_run!(py, keys, "
    //         //     for key in keys:
    //         //         print(key)
    //         // ");
    //         let env = &self.env;
    //         py_run!(py, env, "print(env)");
    //         py_run!(py, env, "print(env._max_episode_steps)");
    //         let dict = self
    //             .env
    //             .getattr(py, "env")?
    //             .getattr(py, "env")?
    //             .getattr(py, "env")?
    //             .getattr(py, "__dict__")?;
    //         py_run!(py, dict, "print('reward_type     = ', dict['reward_type'])");
    //         py_run!(
    //             py,
    //             dict,
    //             "print('continuing_task = ', dict['continuing_task'])"
    //         );
    //         py_run!(
    //             py,
    //             dict,
    //             "print('reset_target    = ', dict['reset_target'])"
    //         );
    //         py_run!(py, dict, "print('spec            = ', dict['spec'])");

    //         // panic!();
    //         Ok(result)
    //     })
    // }
}
