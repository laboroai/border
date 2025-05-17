//! Wrapper of gym environments implemented in Python.
#![allow(clippy::float_cmp)]
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue::Scalar},
    Env, Info, Step,
};
use log::{info, trace};
// use pyo3::IntoPy;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{types::PyModule, PyObject, Python, ToPyObject};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{fmt::Debug, time::Duration};

/// Information given at every step of the interaction with the environment.
///
/// Currently, it is empty and used to match the type signature.
pub struct GymInfo {}

impl Info for GymInfo {}

/// Convert objects for observation and action.
pub trait GymEnvConverter {
    /// Type of observation.
    type Obs: border_core::Obs;

    /// Type of action.
    type Act: border_core::Act;

    /// Configuration.
    type Config: DeserializeOwned + Serialize + Clone + Default;

    /// Convert [`PyObject`] to [`Self::Obs`].
    fn filt_obs(&mut self, obs: PyObject) -> Result<Self::Obs>;

    /// Convert [`Self::Act`] to [`PyObject`].
    fn filt_act(&mut self, act: Self::Act) -> Result<PyObject>;

    /// Creates a converter.
    fn new(config: &Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Called when resetting the environment.
    ///
    /// This method is useful for stateful filters.
    fn reset(&mut self, obs: PyObject) -> Result<Self::Obs> {
        let obs = self.filt_obs(obs)?;
        Ok(obs)
    }
}

/// Configuration of [`GymEnv`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GymEnvConfig<C>
where
    C: GymEnvConverter,
{
    /// The maximum interaction steps in an episode.
    pub max_steps: Option<usize>,

    /// `true` to support rendering for PyBullet gym environment.
    pub pybullet: bool,

    /// Name of the environment, e.g., `CartPole-v0`.
    pub name: String,

    /// Rendering mode, e.g., "human" or "rgb_array".
    pub render_mode: Option<String>,

    /// Wait time at every interaction steps.
    pub wait: Duration,

    /// Converter of observation and action.
    pub converter_config: C::Config,
}

impl<C> Default for GymEnvConfig<C>
where
    C: GymEnvConverter,
{
    fn default() -> Self {
        Self {
            max_steps: None,
            pybullet: false,
            name: "".to_string(),
            render_mode: None,
            wait: Duration::from_millis(0),
            converter_config: Default::default(),
        }
    }
}

impl<C> GymEnvConfig<C>
where
    C: GymEnvConverter,
{
    /// Set `True` when using PyBullet environments.
    pub fn pybullet(mut self, v: bool) -> Self {
        self.pybullet = v;
        self
    }

    /// Set the name of the environment.
    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn render_mode(mut self, render_mode: Option<String>) -> Self {
        self.render_mode = render_mode;
        self
    }

    /// Set wait time in milli seconds.
    pub fn set_wait_in_millis(mut self, millis: u64) -> Self {
        self.wait = Duration::from_millis(millis);
        self
    }

    pub fn converter_config(mut self, config: C::Config) -> Self {
        self.converter_config = config;
        self
    }
}

/// An wrapper of [Gymnasium](https://gymnasium.farama.org).
#[derive(Debug)]
pub struct GymEnv<C>
where
    C: GymEnvConverter,
{
    render: bool,
    env: PyObject,
    count_steps: usize,
    max_steps: Option<usize>,
    converter: C,
    wait: Duration,
    pybullet: bool,
    pybullet_state: Option<PyObject>,
    /// Initial seed.
    ///
    /// This value will be used at the first call of the reset method.
    initial_seed: Option<i64>,
}

impl<C> GymEnv<C>
where
    C: GymEnvConverter,
{
    /// Set rendering mode.
    ///
    /// If `true`, it renders the state at every step.
    pub fn set_render(&mut self, render: bool) -> Result<()> {
        self.render = render;
        if self.pybullet {
            pyo3::Python::with_gil(|py| {
                self.env
                    .call_method(py, "render", ("human",), None)
                    .unwrap();
            });
        }
        Ok(())
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

    // /// Get the number of available actions of atari environments
    // pub fn get_num_actions_atari(&self) -> i64 {
    //     pyo3::Python::with_gil(|py| {
    //         let act_space = self.env.getattr(py, "action_space").unwrap();
    //         act_space.getattr(py, "n").unwrap().extract(py).unwrap()
    //     })
    // }

    /// Returns `is_terminated` and `is_truncated`, extracted from `Step` object in Python.
    fn is_done(step: &PyTuple) -> Result<(i8, i8)> {
        // terminated or truncated
        let is_terminated = match step.get_item(2).extract()? {
            true => 1,
            false => 0,
        };
        let is_truncated = match step.get_item(3).extract()? {
            true => 1,
            false => 0,
        };

        Ok((is_terminated, is_truncated))
    }
}

impl<C> Env for GymEnv<C>
where
    C: GymEnvConverter + Clone,
{
    type Obs = <C as GymEnvConverter>::Obs;
    type Act = <C as GymEnvConverter>::Act;
    type Info = GymInfo;
    type Config = GymEnvConfig<C>;

    /// Resets the environment and returns an observation.
    ///
    /// This method also resets the converter of type `C`.
    ///
    /// In this environment, `is_done` should be None.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
        trace!("PyGymEnv::reset()");
        assert_eq!(is_done, None);

        // Initial observation
        let ret = pyo3::Python::with_gil(|py| {
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
                let floor: &PyModule = self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
                floor.getattr("add_floor")?.call1((&self.env,)).unwrap();
            }
            // Reset the state
            Ok(self.converter.reset(obs)?)
        });

        // Rendering
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
    fn step(&mut self, act: &Self::Act) -> (Step<Self>, Record) {
        trace!("PyGymEnv::step()");

        pyo3::Python::with_gil(|py| {
            if self.render {
                if !self.pybullet {
                    let _ = self.env.call_method0(py, "render");
                } else {
                    let cam: &PyModule = self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
                    cam.getattr("update_camera_pos")
                        .unwrap()
                        .call1((&self.env,))
                        .unwrap();
                }
                std::thread::sleep(self.wait);
            }

            // Run a step
            let step_py = {
                let a_py = self.converter.filt_act(act.clone()).unwrap();
                self.env.call_method(py, "step", (a_py,), None).unwrap()
            };
            let step: &PyTuple = step_py.extract(py).unwrap();

            // Observation at the next step
            let obs = {
                let obs_py = step.get_item(0).to_owned();
                self.converter.filt_obs(obs_py.to_object(py)).unwrap()
                // self.converter.filt_obs(obs_py.into()).unwrap()
            };

            // Reward
            let reward: Vec<f32> = vec![step.get_item(1).extract().unwrap()];

            // Terminated/Truncated flags
            let (is_terminated, mut is_truncated) = {
                let (is_terminated, is_truncated) = Self::is_done(step).unwrap();
                (vec![is_terminated], vec![is_truncated])
            };

            // Misc.
            let mut record = Record::empty();
            let info = GymInfo {};
            let init_obs = None;

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

            // Returned step object
            let step = Step {
                obs,
                act: act.clone(),
                reward,
                is_terminated,
                is_truncated,
                info,
                init_obs,
            };

            (step, record)
        })
    }

    /// Creates [`GymEnv`].
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

        // For some unknown reason, Mujoco requires this import
        if py.import("IPython").is_ok() {}

        let name = config.name.as_str();
        let (env, render) = if !config.pybullet {
            let gym = py.import("f32_wrapper")?;
            let render = config.render_mode.is_some();
            let env = {
                let kwargs = if let Some(render_mode) = config.render_mode.clone() {
                    Some(vec![("render_mode", render_mode)].into_py_dict(py))
                } else {
                    None
                };
                gym.getattr("make_f32")?.call((name,), kwargs)?
            };

            (env, render)
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
        println!("Action space = {:?}", action_space);
        let observation_space = env.getattr("observation_space")?;
        println!("Observation space = {:?}", observation_space);

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
            converter: C::new(&config.converter_config)?,
            render,
            count_steps: 0,
            wait: config.wait,
            max_steps: config.max_steps,
            pybullet: config.pybullet,
            pybullet_state,
            initial_seed: Some(seed),
        })
    }
}
