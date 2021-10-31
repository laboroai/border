//! Configuration of [BorderAtariEnv](super::BorderAtariEnv).
//!
//! If environment variable `ATARI_ROM_DIR` exists, it is used as the directory
//! from which ROM images of the Atari games is loaded.
use std::{env, default::Default};
use border_core::{Obs, Act};
use super::{BorderAtariObsFilter, BorderAtariActFilter};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
/// Configuration of [BorderAtariEnv](super::BorderAtariEnv).
pub struct BorderAtariEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    pub(super) rom_dir: String,
    pub(super) name: String,
    pub(super) obs_filter_config: OF::Config,
    pub(super) act_filter_config: AF::Config,
}

impl<O, A, OF, AF> Default for BorderAtariEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    fn default() -> Self {
        let rom_dir = if let Ok(var) = env::var("ATARI_ROM_DIR") {
            var
        } else {
            "".to_string()
        };

        Self {
            rom_dir,
            name: "".to_string(),
            obs_filter_config: Default::default(),
            act_filter_config: Default::default(),
        }
    }
}

impl<O, A, OF, AF> BorderAtariEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    /// Sets the name of the game.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}
