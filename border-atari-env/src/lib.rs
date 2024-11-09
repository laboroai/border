//! A thin wrapper of [`atari-env`](https://crates.io/crates/atari-env) for [`Border`](https://crates.io/crates/border).
//!
//! The code under [atari_env] is adapted from the
//! [`atari-env`](https://crates.io/crates/atari-env) crate
//! (rev = `0ef0422f953d79e96b32ad14284c9600bd34f335`),
//! because the crate registered in crates.io does not implement
//! [`atari_env::AtariEnv::lives()`] method, which is required for episodic life environments.
//!
//! This environment applies some preprocessing to observation as in
//! [`atari_wrapper.py`](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
//!
//! You need to place Atari Rom directories under the directory specified by environment variable
//! `ATARI_ROM_DIR`. An easy way to do this is to use [AutoROM](https://pypi.org/project/AutoROM/)
//! Python package.
//!
//! ```bash
//! pip install autorom
//! mkdir $HOME/atari_rom
//! AutoROM --install-dir $HOME/atari_rom
//! export ATARI_ROM_DIR=$HOME/atari_rom
//! ```
//!
//! Here is an example of running Pong environment with a random policy.
//!
//! ```no_run
//! use anyhow::Result;
//! use border_atari_env::{
//!     BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig,
//!     BorderAtariObs, BorderAtariObsRawFilter,
//! };
//! use border_core::{Env as _, Policy, DefaultEvaluator, Evaluator as _};
//!
//! # type Obs = BorderAtariObs;
//! # type Act = BorderAtariAct;
//! # type ObsFilter = BorderAtariObsRawFilter<Obs>;
//! # type ActFilter = BorderAtariActRawFilter<Act>;
//! # type EnvConfig = BorderAtariEnvConfig<Obs, Act, ObsFilter, ActFilter>;
//! # type Env = BorderAtariEnv<Obs, Act, ObsFilter, ActFilter>;
//! #
//! # #[derive(Clone)]
//! # struct RandomPolicyConfig {
//! #     pub n_acts: usize,
//! # }
//! #
//! # struct RandomPolicy {
//! #     n_acts: usize,
//! # }
//! #
//! # impl RandomPolicy {
//! #     pub fn build(n_acts: usize) -> Self {
//! #         Self { n_acts }
//! #     }
//! # }
//! #
//! # impl Policy<Env> for RandomPolicy {
//! #     fn sample(&mut self, _: &Obs) -> Act {
//! #         fastrand::u8(..self.n_acts as u8).into()
//! #     }
//! # }
//! #
//! # fn env_config(name: String) -> EnvConfig {
//! #     EnvConfig::default().name(name)
//! # }
//! #
//! fn main() -> Result<()> {
//! #     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
//! #     fastrand::seed(42);
//! #
//!     // Creates Pong environment
//!     let env_config = env_config("pong".to_string());
//!
//!     // Creates a random policy
//!     let n_acts = 4;
//!     let mut policy = Box::new(RandomPolicy::build(n_acts));
//!
//!     // Runs evaluation
//!     let env_config = env_config.render(true);
//!     let _ = DefaultEvaluator::new(&env_config, 0, 5)?.evaluate(&mut policy);
//!
//!     Ok(())
//! }
//! ```
//! [`atari_env::AtariEnv::lives()`]: atari_env::AtariEnv::lives
mod act;
pub mod atari_env;
mod env;
mod obs;
pub mod util;
pub use act::{BorderAtariAct, BorderAtariActFilter, BorderAtariActRawFilter};
pub use env::{BorderAtariEnv, BorderAtariEnvConfig};
pub use obs::{BorderAtariObs, BorderAtariObsFilter, BorderAtariObsRawFilter};
