//! A thin wrapper of [atari-env](https://crates.io/crates/atari-env).
//!
//! The code under [atari_env] is adapted from the
//! [atari-env](https://crates.io/crates/atari-env) crate
//! (rev = `0ef0422f953d79e96b32ad14284c9600bd34f335`),
//! because the crate registered in crates.io does not implement
//! [atari_env::AtariEnv]`::lives()` method, which is required for episodic life.
//!
//! This environment applies some preprocessing to observation as in
//! [atari_wrapper.py](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
mod act;
mod obs;
mod env;
pub mod util;
pub mod atari_env;
pub use act::{BorderAtariAct, BorderAtariActFilter, BorderAtariActRawFilter};
pub use obs::{BorderAtariObs, BorderAtariObsFilter, BorderAtariObsRawFilter};
pub use env::{BorderAtariEnv, BorderAtariEnvConfig};
