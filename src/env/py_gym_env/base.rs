//! Wrapper of gym environments implemented in Python.
#![allow(clippy::float_cmp)]
use log::trace;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{PyObject, PyResult, Python, ToPyObject, types::PyModule};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::{error::Error, fmt::Debug, time::Duration};

use super::AtariWrapper;
use crate::core::{record::Record, Act, Env, Info, Obs, Step};

/// Information given at every step of the interaction with the environment.
///
/// Currently, it is empty and used to match the type signature.
pub struct PyGymInfo {}

impl Info for PyGymInfo {}

/// Shape of observation or action.
pub trait Shape: Clone + Debug {
    /// Returns the shape of Shape of observation or action.
    ///
    /// This trait is used for conversion of PyObject in [`super::obs::pyobj_to_arrayd`] and
    fn shape() -> &'static [usize];

    /// Returns `true` if you would like to squeeze the first dimension of the array
    /// before conversion into an numpy array in Python. The first dimension may
    /// correspond to process indices for vectorized environments.
    /// This method is used in
    /// [`super::act_c::to_pyobj`] and [`super::act_c::PyGymEnvContinuousActRawFilter::filt`].
    fn squeeze_first_dim() -> bool {
        false
    }
}

/// Convert PyObject to PyGymEnv::Obs.
pub trait PyGymEnvObsFilter<O: Obs> {
    /// Convert PyObject into observation with filtering.
    fn filt(&mut self, obs: PyObject) -> (O, Record);

    /// Called when resetting the environment.
    ///
    /// This method is useful for stateful filters.
    fn reset(&mut self, obs: PyObject) -> O {
        let (obs, _) = self.filt(obs);
        obs
    }
}

/// Convert PyGymEnv::Act to PyObject.
///
/// This trait should support vectorized environments.
pub trait PyGymEnvActFilter<A: Act> {
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
}

/// Constructs [PyGymEnv].
pub struct PyGymEnvBuilder<O, A, OF, AF> {
    max_steps: Option<usize>,
    atari_wrapper: Option<AtariWrapper>,
    pybullet: bool,
    phantom: PhantomData<(O, A, OF, AF)>,
}

impl<O, A, OF, AF> Default for PyGymEnvBuilder<O, A, OF, AF> {
    fn default() -> Self {
        Self {
            max_steps: None,
            atari_wrapper: None,
            pybullet: false,
            phantom: PhantomData,
        }
    }
}

impl<O, A, OF, AF> PyGymEnvBuilder<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    /// Set `True` when using PyBullet environments.
    pub fn pybullet(mut self, v: bool) -> Self {
        self.pybullet = v;
        self
    }

    /// Set `True` when using Atari wrapper.
    pub fn atari_wrapper(mut self, v: Option<AtariWrapper>) -> Self {
        self.atari_wrapper = v;
        self
    }

    /// Constructs [PyGymEnv].
    pub fn build(
        self,
        name: &str,
        obs_filter: OF,
        act_filter: AF,
    ) -> PyResult<PyGymEnv<O, A, OF, AF>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // sys.argv is used by pyglet library, which is responsible for rendering.
        // Depending on the python interpreter, however, sys.argv can be empty.
        // For that case, sys argv is set here.
        // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
        let locals = [("sys", py.import("sys")?)].into_py_dict(py);
        let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

        // import pybullet-gym if it exists
        if py.import("pybulletgym").is_ok() {}

        let env = if let Some(mode) = self.atari_wrapper {
            let mode = match mode {
                AtariWrapper::Train => true,
                AtariWrapper::Eval => false,
            };
            let gym = py.import("atari_wrappers")?;
            let env = gym.call("make_env_single_proc", (name, true, mode), None)?;
            env.call_method("seed", (42,), None)?;
            env
        } else {
            let gym = py.import("gym")?;
            let env = gym.call("make", (name,), None)?;
            env.call_method("seed", (42,), None)?;
            env
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
        let observation_space = observation_space.getattr("shape")?.extract()?;

        let pybullet_state = if !self.pybullet {
            None
        } else {
            let pybullet_state = Python::with_gil(|py| {
                PyModule::from_code(py, r#"
_torsoId = None
_floor = False

def add_floor(env):
    global _floor
    if not _floor:
        p = env.env._p
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        _floor = True
        env.env.stateId = p.saveState()

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
    p = env.env._p
    torsoId = get_torso_id(env.env._p)
    if torsoId >= 0:
        distance = 5
        yaw = 0
        humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
        p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            "#, "pybullet_state.py", "pybullet_state").unwrap().to_object(py)
            });
            Some(pybullet_state)
        };

        Ok(PyGymEnv {
            env: env.into(),
            action_space,
            observation_space,
            // TODO: consider remove RefCell, raw value instead
            obs_filter,
            act_filter,
            render: false,
            count_steps: RefCell::new(0),
            wait_in_render: Duration::from_millis(0),
            max_steps: self.max_steps,
            pybullet: self.pybullet,
            pybullet_state,
            phantom: PhantomData,
        })
    }
}

/// Represents an environment in [OpenAI gym](https://github.com/openai/gym).
/// The code is adapted from [tch-rs RL example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/reinforcement-learning).
#[derive(Debug)]
pub struct PyGymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    render: bool,
    env: PyObject,
    action_space: i64,
    observation_space: Vec<usize>,
    count_steps: RefCell<usize>,
    max_steps: Option<usize>,
    obs_filter: OF,
    act_filter: AF,
    wait_in_render: Duration,
    pybullet: bool,
    pybullet_state: Option<PyObject>,
    phantom: PhantomData<(O, A)>,
}

impl<O, A, OF, AF> PyGymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    /// Constructs an environment.
    ///
    /// `name` is the name of the environment, which is implemented in OpenAI gym.
    pub fn new(
        name: &str,
        obs_filter: OF,
        act_filter: AF,
        atari_wrapper: Option<AtariWrapper>,
    ) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // sys.argv is used by pyglet library, which is responsible for rendering.
        // Depending on the python interpreter, however, sys.argv can be empty.
        // For that case, sys argv is set here.
        // See https://github.com/PyO3/pyo3/issues/1241#issuecomment-715952517
        let locals = [("sys", py.import("sys")?)].into_py_dict(py);
        let _ = py.eval("sys.argv.insert(0, 'PyGymEnv')", None, Some(&locals))?;

        // import pybullet-gym if it exists
        if py.import("pybulletgym").is_ok() {}

        let env = if let Some(mode) = atari_wrapper {
            let mode = match mode {
                AtariWrapper::Train => true,
                AtariWrapper::Eval => false,
            };
            let gym = py.import("atari_wrappers")?;
            let env = gym.call("make_env_single_proc", (name, true, mode), None)?;
            env.call_method("seed", (42,), None)?;
            env
        } else {
            let gym = py.import("gym")?;
            let env = gym.call("make", (name,), None)?;
            env.call_method("seed", (42,), None)?;
            env
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
        let observation_space = observation_space.getattr("shape")?.extract()?;

        Ok(PyGymEnv {
            render: false,
            env: env.into(),
            action_space,
            observation_space,
            // TODO: consider remove RefCell, raw value instead
            count_steps: RefCell::new(0),
            max_steps: None,
            obs_filter,
            act_filter,
            wait_in_render: Duration::from_millis(0),
            pybullet: false,
            pybullet_state: None,
            phantom: PhantomData,
        })
    }

    /// Set rendering mode.
    ///
    /// If `true`, it renders the state at every step.
    pub fn set_render(&mut self, render: bool) {
        self.render = render;
        if self.pybullet {
            pyo3::Python::with_gil(|py| {
                self.env.call_method0(py, "render").unwrap();
            });
        }
    }

    /// Set the maximum number of steps in the environment.
    pub fn max_steps(mut self, v: Option<usize>) -> Self {
        self.max_steps = v;
        self
    }

    /// Set time for sleep in rendering.
    pub fn set_wait_in_render(&mut self, d: Duration) {
        self.wait_in_render = d;
    }

    /// Get the number of available actions of atari environments
    pub fn get_num_actions_atari(&self) -> i64 {
        pyo3::Python::with_gil(|py| {
            let act_space = self.env.getattr(py, "action_space").unwrap();
            act_space.getattr(py, "n").unwrap().extract(py).unwrap()
        })
    }
}

impl<O, A, OF, AF> Env for PyGymEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act + Debug,
    OF: PyGymEnvObsFilter<O>,
    AF: PyGymEnvActFilter<A>,
{
    type Obs = O;
    type Act = A;
    type Info = PyGymInfo;

    /// Resets the environment, the obs/act filters and returns the observation tensor.
    ///
    /// In this environment, the length of `is_done` is assumed to be 1.
    ///
    /// TODO: defines appropriate error for the method and returns it.
    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<O, Box<dyn Error>> {
        trace!("PyGymEnv::reset()");

        // Reset the action filter, required for stateful filters.
        self.act_filter.reset(&is_done);

        // Reset the environment
        let reset = match is_done {
            None => true,
            Some(v) => {
                debug_assert_eq!(v.len(), 1);
                v[0] != 0
            }
        };

        if !reset {
            Ok(O::dummy(1))
        } else {
            pyo3::Python::with_gil(|py| {
                let obs = self.env.call_method0(py, "reset")?;
                if self.pybullet && self.render {
                    let floor: &PyModule = self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
                    floor.call1("add_floor", (&self.env,)).unwrap();
                }
                Ok(self.obs_filter.reset(obs))
            })
        }
    }

    /// Runs a step of the environment's dynamics.
    ///
    /// It returns [`Step`] and [`Record`] objects.
    /// The [`Record`] is composed of [`Record`]s constructed in [`PyGymEnvObsFilter`] and
    /// [`PyGymEnvActFilter`].
    fn step(&mut self, a: &A) -> (Step<Self>, Record) {
        trace!("PyGymEnv::step()");

        pyo3::Python::with_gil(|py| {
            if self.render {
                if !self.pybullet {
                    let _ = self.env.call_method0(py, "render");
                }
                else {
                    let cam: &PyModule = self.pybullet_state.as_ref().unwrap().extract(py).unwrap();
                    cam.call1("update_camera_pos", (&self.env,)).unwrap();
                }
                std::thread::sleep(self.wait_in_render);
            }

            let (a_py, record_a) = self.act_filter.filt(a.clone());
            let ret = self.env.call_method(py, "step", (a_py,), None).unwrap();
            let step: &PyTuple = ret.extract(py).unwrap();
            let obs = step.get_item(0).to_owned();
            let (obs, record_o) = self.obs_filter.filt(obs.to_object(py));
            let reward: Vec<f32> = vec![step.get_item(1).extract().unwrap()];
            let mut is_done: Vec<i8> = vec![if step.get_item(2).extract().unwrap() {
                1
            } else {
                0
            }];

            let c = *self.count_steps.borrow();
            self.count_steps.replace(c + 1);
            if let Some(max_steps) = self.max_steps {
                if *self.count_steps.borrow() >= max_steps {
                    is_done[0] = 1;
                    self.count_steps.replace(0);
                }
            };

            (
                Step::<Self>::new(obs, a.clone(), reward, is_done, PyGymInfo {}),
                record_o.merge(record_a),
            )
        })
    }
}
