//! A thin wrapper of [atari-env](https://crates.io/crates/atari-env).
//!
//! This applies some preprocessing to observation as in
//! [atari_wrapper.py](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
mod act;
mod obs;
mod env;
pub mod util;
pub use act::{BorderAtariAct, BorderAtariActFilter, BorderAtariActRawFilter};
pub use obs::{BorderAtariObs, BorderAtariObsFilter, BorderAtariObsRawFilter};
pub use env::{BorderAtariEnv, BorderAtariEnvConfig};
