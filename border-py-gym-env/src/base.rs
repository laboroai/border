//! Wrapper of gym environments implemented in Python.
#![allow(clippy::float_cmp)]
use crate::{AtariWrapper, GymEnvConfig};
use anyhow::Result;
use border_core::{record::Record, Act, Env, Info, Obs, Step};
use log::{info, trace};
// use pyo3::IntoPy;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{types::PyModule, PyObject, Python, ToPyObject};
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::{fmt::Debug, time::Duration};

/// Information given at every step of the interaction with the environment.
///
/// Currently, it is empty and used to match the type signature.
pub struct GymInfo {}

impl Info for GymInfo {}

/// Convert [PyObject] to [PyGymEnv]::Obs with a preprocessing.
pub trait GymObsFilter<O: Obs> {
    /// Configuration.
    type Config: Clone + Default + Serialize + DeserializeOwned;

    /// Build filter.
    fn build(config: &Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Convert PyObject into observation with filtering.
    fn filt(&mut self, obs: PyObject) -> (O, Record);

    /// Called when resetting the environment.
    ///
    /// This method is useful for stateful filters.
    fn reset(&mut self, obs: PyObject) -> O {
        let (obs, _) = self.filt(obs);
        obs
    }

    /// Returns default configuration.
    fn default_config() -> Self::Config {
        Self::Config::default()
    }
}

/// Convert [PyGymEnv]::Act to [PyObject] with a preprocessing.
///
/// This trait should support vectorized environments.
pub trait GymActFilter<A: Act> {
    /// Configuration.
    type Config: Clone + Default + Serialize + DeserializeOwned;

    /// Build filter.
    fn build(config: &Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Filter action and convert it to PyObject.
    ///
    /// For vectorized environments, `act` should have actions for all environments in
    /// the vectorized environment. The return values will be a `PyList` object, each
    /// element is an action of the corresponding environment.
    fn filt(&mut self, act: A) -> (PyObject, Record);

    /// Called when resetting the environment.
    ///
    /// This method is useful for stateful filters.
    /// This method support vectorized environment
    fn reset(&mut self, _is_done: &Option<&Vec<i8>>) {}

    /// Returns default configuration.
    fn default_config() -> Self::Config {
        Self::Config::default()
    }
}

/// An environment in [OpenAI gym](https://github.com/openai/gym).
#[derive(Debug)]
pub struct GymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    render: bool,

    env: PyObject,

    #[allow(dead_code)]
    action_space: i64,

    #[allow(dead_code)]
    observation_space: Vec<usize>,

    count_steps: usize,

    max_steps: Option<usize>,

    obs_filter: OF,

    act_filter: AF,

    wait: Duration,

    pybullet: bool,

    pybullet_state: Option<PyObject>,

    /// Initial seed.
    ///
    /// This value will be used at the first call of the reset method.
    initial_seed: Option<i64>,

    phantom: PhantomData<(O, A)>,
}

impl<O, A, OF, AF> GymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    /// Set rendering mode.
    ///
    /// If `true`, it renders the state at every step.
    pub fn set_render(&mut self, render: bool) {
        self.render = render;
        if self.pybullet {
            pyo3::Python::with_gil(|py| {
                // self.env.call_method0(py, "render").unwrap();
                self.env
                    .call_method(py, "render", ("human",), None)
                    .unwrap();
            });
        }
    }

    /// Set the maximum number of steps in the environment.
    pub fn max_steps(mut self, v: Option<usize>) -> Self {
        self.max_steps = v;
        self
    }

    /// Set wait time at every interaction steps.
    pub fn set_wait(&mut self, d: Duration) {
        self.wait = d;
    }

    /// Get the number of available actions of atari environments
    pub fn get_num_actions_atari(&self) -> i64 {
        pyo3::Python::with_gil(|py| {
            let act_space = self.env.getattr(py, "action_space").unwrap();
            act_space.getattr(py, "n").unwrap().extract(py).unwrap()
        })
    }
}

impl<O, A, OF, AF> Env for GymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act + Debug,
    OF: GymObsFilter<O>,
    AF: GymActFilter<A>,
{
    type Obs = O;
    type Act = A;
    type Info = GymInfo;
    type Config = GymEnvConfig<O, A, OF, AF>;

    /// Currently it supports non-vectorized environment.
    fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record)
    where
        Self: Sized,
    {
        let (step, record) = self.step(a);
        assert_eq!(step.is_done.len(), 1);
        let step = if step.is_done[0] == 1 {
            let init_obs = self.reset(None).unwrap();
            Step {
                act: step.act,
                obs: step.obs,
                reward: step.reward,
                is_done: step.is_done,
                info: step.info,
                init_obs,
            }
        } else {
            step
        };

        (step, record)
    }

    /// Resets the environment and returns an observation.
    ///
    /// This method also resets the [`PyGymObsFilter`] adn [`PyGymActFilter`].
    ///
    /// In this environment, the length of `is_done` is assumed to be 1.
    ///
    /// [`PyGymObsFilter`]: crate::PyGymEnvObsFilter
    /// [`PyGymActFilter`]: crate::PyGymEnvActFilter
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<O> {
        trace!("PyGymEnv::reset()");

        // Reset the action filter, required for stateful filters.
        self.act_filter.reset(&is_done);

        // Reset the environment
        let reset = match is_done {
            None => true,
            // when reset() is called in border_core::util::sample()
            Some(v) => {
                debug_assert_eq!(v.len(), 1);
                v[0] != 0
            }
        };

        let ret = if !reset {
            Ok(O::dummy(1))
        } else {
            pyo3::Python::with_gil(|py| {
                let obs = {
                    let ret_values = if let Some(seed) = self.initial_seed {
                        self.initial_seed = None;
                        let kwargs = match self.pybullet {
                            true => None,
                            false => Some(vec![("seed", seed)].into_py_dict(py)),
                        };
                        self.env.call_method(py, "reset", (), kwargs)?
                    } else {
                        self.env.call_method0(py, "reset")?
                    };
                    let ret_values_: &PyTuple = ret_values.extract(py).unwrap();
                    ret_values_.get_item(0).extract().unwrap()
                };

                if self.pybullet && self.render {
                    let floor: &PyModule =
                        self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
                    floor.getattr("add_floor")?.call1((&self.env,)).unwrap();
                }
                Ok(self.obs_filter.reset(obs))
            })
        };

        if self.pybullet && self.render {
            pyo3::Python::with_gil(|py| {
                // self.env.call_method0(py, "render").unwrap();
                self.env
                    .call_method(py, "render", ("human",), None)
                    .unwrap();
            });
        }

        ret
    }

    /// Resets the environment with the given index.
    ///
    /// Specifically, env.reset(seed=ix) is called in the Python interpreter.
    fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs> {
        self.initial_seed = Some(ix as _);
        self.reset(None)
    }

    /// Runs a step of the environment's dynamics.
    ///
    /// It returns [`Step`] and [`Record`] objects.
    /// The [`Record`] is composed of [`Record`]s constructed in [`PyGymEnvObsFilter`] and
    /// [`PyGymEnvActFilter`].
    fn step(&mut self, a: &A) -> (Step<Self>, Record) {
        fn is_done(step: &PyTuple) -> i8 {
            // terminated or truncated
            let terminated: bool = step.get_item(2).extract().unwrap();
            let truncated: bool = step.get_item(3).extract().unwrap();

            if terminated | truncated {
                1
            } else {
                0
            }
        }

        trace!("PyGymEnv::step()");

        pyo3::Python::with_gil(|py| {
            if self.render {
                if !self.pybullet {
                    let _ = self.env.call_method0(py, "render");
                } else {
                    let cam: &PyModule = self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
                    // cam.call1("update_camera_pos", (&self.env,)).unwrap();
                    cam.getattr("update_camera_pos")
                        .unwrap()
                        .call1((&self.env,))
                        .unwrap();
                }
                std::thread::sleep(self.wait);
            }

            let (a_py, record_a) = self.act_filter.filt(a.clone());
            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();
            let obs = step.get_item(0).to_owned();
            let (obs, record_o) = self.obs_filter.filt(obs.to_object(py));
            let reward: Vec<f32> = vec![step.get_item(1).extract().unwrap()];
            let mut is_done: Vec<i8> = vec![is_done(step)];

            // let c = *self.count_steps.borrow();
            self.count_steps += 1; //.replace(c + 1);
            if let Some(max_steps) = self.max_steps {
                if self.count_steps >= max_steps {
                    is_done[0] = 1;
                    self.count_steps = 0;
                }
            };

            (
                Step::<Self>::new(obs, a.clone(), reward, is_done, GymInfo {}, O::dummy(1)),
                record_o.merge(record_a),
            )
        })
    }

    /// Constructs [`PyGymEnv`].
    ///
    /// * `seed` - The seed value of the random number generator.
    ///   This value will be used at the first call of the reset method.
    fn build(config: &Self::Config, seed: i64) -> Result<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // sys.argv is used by pyglet library, which is responsible for rendering.
        // Depending on the python interpreter, however, sys.argv can be empty.
        // For that case, sys argv is set here.
        // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
        let locals = [("sys", py.import("sys")?)].into_py_dict(py);
        let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;
        let path = py.eval("sys.path", None, Some(&locals)).unwrap();
        let ver = py.eval("sys.version", None, Some(&locals)).unwrap();
        info!("Initialize PyGymEnv");
        info!("{}", path);
        info!("Python version = {}", ver);

        // import pybullet-gym if it exists
        if py.import("pybulletgym").is_ok() {}

        let name = config.name.as_str();
        let (env, render) = if let Some(mode) = config.atari_wrapper.as_ref() {
            let mode = match mode {
                AtariWrapper::Train => true,
                AtariWrapper::Eval => false,
            };
            let gym = py.import("atari_wrappers")?;
            let env = gym
                .getattr("make_env_single_proc")?
                .call((name, true, mode), None)?;
            (env, false)
        } else if !config.pybullet {
            let gym = py.import("f32_wrapper")?;
            let kwargs = if let Some(render_mode) = config.render_mode.clone() {
                Some(vec![("render_mode", render_mode)].into_py_dict(py))
            } else {
                None
            };
            let env = gym.getattr("make_f32")?.call((name,), kwargs)?;
            (env, false)
        } else {
            let gym = py.import("f32_wrapper")?;
            let kwargs = None;
            let env = gym.getattr("make_f32")?.call((name,), kwargs)?;
            if config.render_mode.is_some() {
                env.call_method("render", ("human",), None).unwrap();
                (env, true)
            } else {
                (env, false)
            }
        };

        // TODO: consider removing action_space and observation_space.
        // Act/obs types are specified by type parameters.
        let action_space = env.getattr("action_space")?;
        let action_space = if let Ok(val) = action_space.getattr("n") {
            val.extract()?
        } else {
            let action_space: Vec<i64> = action_space.getattr("shape")?.extract()?;
            action_space[0]
        };
        let observation_space = env.getattr("observation_space")?;
        println!("Observation space = {:?}", observation_space);
        let observation_space = observation_space.getattr("shape")?.extract()?;

        let pybullet_state = if !config.pybullet {
            None
        } else {
            let pybullet_state = Python::with_gil(|py| {
                PyModule::from_code(
                    py,
                    r#"
_torsoId = None
_floor = False

def unwrap(env):
    while True:
        if hasattr(env, "_p"):
            return env
        else:
            env = env.env

def add_floor(env):
    global _floor
    if not _floor:
        env = unwrap(env)
        p = env._p
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        _floor = True
        env.stateId = p.saveState()

def get_torso_id(p):
    global _torsoId
    if _torsoId is None:
        torsoId = -1
        for i in range(p.getNumBodies()):
            print(p.getBodyInfo(i))
            if p.getBodyInfo(i)[0].decode() == "torso":
                torsoId = i
                print("found torso")
        _torsoId = torsoId
    
    return _torsoId

def update_camera_pos(env):
    env = unwrap(env)
    p = env._p
    torsoId = get_torso_id(p)
    if torsoId >= 0:
        distance = 5
        yaw = 0
        humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
        p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            "#,
                    "pybullet_state.py",
                    "pybullet_state",
                )
                .unwrap()
                .to_object(py)
            });
            Some(pybullet_state)
        };

        Ok(GymEnv {
            env: env.into(),
            action_space,
            observation_space,
            obs_filter: OF::build(&config.obs_filter_config.as_ref().unwrap())?,
            act_filter: AF::build(&config.act_filter_config.as_ref().unwrap())?,
            render,
            count_steps: 0,
            wait: config.wait.clone(),
            max_steps: config.max_steps,
            pybullet: config.pybullet,
            pybullet_state,
            initial_seed: Some(seed),
            phantom: PhantomData,
        })
    }
}
