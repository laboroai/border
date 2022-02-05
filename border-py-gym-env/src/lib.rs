//! A wrapper of [gym](https://gym.openai.com) environments on Python.
//!
//! [PyGymEnv] is a wrapper of [gym](https://gym.openai.com) based on [PyO3](https://github.com/PyO3/pyo3).
//! It supports some [classic control](https://gym.openai.com/envs/#classic_control),
//! [Atari](https://gym.openai.com/envs/#atari), and [PyBullet](https://github.com/benelot/pybullet-gym)
//! environments.
//!
//! This wrapper accepts array-like observation and action
//! ([Box](https://github.com/openai/gym/blob/master/gym/spaces/box.py) spaces), and
//! discrete action. In order to interact with Python interpreter where gym is running,
//! [PyGymEnvObsFilter] and [PyGymEnvActFilter] provides interfaces for converting Python object
//! (numpy array) to/from ndarrays in Rust. [PyGymEnvObsRawFilter],
//! [PyGymEnvContinuousActRawFilter] and [PyGymEnvDiscreteActRawFilter] do the conversion for environments
//! where observation and action are arrays. In addition to the data conversion between Python and Rust,
//! we can implements arbitrary preprocessing in these filters. For example, [FrameStackFilter] keeps
//! four consevutive observation frames (images) and outputs a stack of these frames.
//!
//! For Atari environments, a tweaked version of
//! [atari_wrapper.py](https://github.com/taku-y/border/blob/main/examples/atari_wrappers.py)
//! is required to be in `PYTHONPATH`. The frame stacking preprocessing is implemented in
//! [FrameStackFilter] as an [PyGymEnvObsFilter].
//!
//! Examples with a random controller ([Policy](border_core::Policy)) are in
//! [examples](https://github.com/taku-y/border/blob/main/border-py-gym-env/examples) directory.
//! Examples with `border-tch-agents`, which are collections of RL agents implemented with tch-rs,
//! are in [here](https://github.com/taku-y/border/blob/main/border/examples).
mod act_c;
mod act_d;
mod atari;
mod base;
mod config;
mod obs;
mod vec;
pub use act_c::{to_pyobj, PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter};
pub use act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter};
pub use atari::AtariWrapper;
pub use base::{PyGymEnv, PyGymEnvActFilter, PyGymEnvObsFilter, PyGymInfo};
pub use config::PyGymEnvConfig;
pub use obs::{pyobj_to_arrayd, FrameStackFilter, PyGymEnvObs, PyGymEnvObsRawFilter};
// pub use vec::{PyVecGymEnv, PyVecGymEnvConfig};
