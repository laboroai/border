//! A wrapper of [Gymnasium](https://gymnasium.farama.org) environments on Python.
//!
//! [`GymEnv`] is a wrapper of [Gymnasium](https://gymnasium.farama.org) based on [`PyO3`](https://github.com/PyO3/pyo3).
//! It has been tested on some of [classic control](https://gymnasium.farama.org/environments/classic_control/) and
//! [Gymnasium-Robotics](https://robotics.farama.org) environments.
//!
//! ```note
//! In a past, [`Atari`](https://gym.openai.com/envs/#atari), and
//! [`PyBullet`](https://github.com/benelot/pybullet-gym) environments were supported.
//! However, currently they are not tested.
//! ```
//!
//! This wrapper accepts array-like observation and action
//! ([`Box`](https://github.com/openai/gym/blob/master/gym/spaces/box.py) spaces), and
//! discrete action. In order to interact with Python interpreter where gym is running,
//! [`GymObsFilter`] and [`GymActFilter`] provides interfaces for converting Python object
//! (numpy array) to/from ndarrays in Rust. [`GymObsFilter`],
//! [`ContinuousActFilter`] and [`DiscreteActFilter`] do the conversion for environments
//! where observation and action are arrays. In addition to the data conversion between Python and Rust,
//! we can implements arbitrary preprocessing in these filters. For example, [`FrameStackFilter`] keeps
//! four consevutive observation frames (images) and outputs a stack of these frames.
//!
//! For Atari environments, a tweaked version of
//! [`atari_wrapper.py`](https://github.com/taku-y/border/blob/main/examples/atari_wrappers.py)
//! is required to be in `PYTHONPATH`. The frame stacking preprocessing is implemented in
//! [`FrameStackFilter`] as an [`GymObsFilter`].
//!
//! Examples with a random controller ([`Policy`](border_core::Policy)) are in
//! [`examples`](https://github.com/taku-y/border/blob/main/border-py-gym-env/examples) directory.
//! Examples with `border-tch-agents`, which are collections of RL agents implemented with tch-rs,
//! are in [here](https://github.com/taku-y/border/blob/main/border/examples).
mod act;
mod act_c;
mod act_d;
mod atari;
mod base;
mod config;
mod obs;
pub mod util;
mod vec;
pub use act::{
    ContinuousActFilter, ContinuousActFilterConfig, DiscreteActFilter, DiscreteActFilterConfig,
};
pub use act_c::{to_pyobj, GymContinuousAct};
pub use act_d::GymDiscreteAct;
pub use atari::AtariWrapper;
pub use base::{GymActFilter, GymEnv, GymInfo, GymObsFilter};
pub use config::GymEnvConfig;
#[allow(deprecated)]
pub use obs::{
    ArrayDictObsFilter, ArrayDictObsFilterConfig, ArrayObsFilter, ArrayObsFilterConfig,
    FrameStackFilter, FrameStackFilterConfig, GymObs,
};
// pub use vec::{PyVecGymEnv, PyVecGymEnvConfig};
