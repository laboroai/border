//! Configuration of [BorderAtariEnv](super::BorderAtariEnv).
//!
//! If environment variable `ATARI_ROM_DIR` exists, it is used as the directory
//! from which ROM images of the Atari games is loaded.
use super::{BorderAtariActFilter, BorderAtariObsFilter};
use border_core::{Act, Obs};
use serde::{Deserialize, Serialize};
use std::{default::Default, env};

#[derive(Serialize, Deserialize, Debug)]
/// Configuration of [`BorderAtariEnv`](super::BorderAtariEnv).
pub struct BorderAtariEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    pub rom_dir: String,
    pub name: String,
    pub obs_filter_config: OF::Config,
    pub act_filter_config: AF::Config,
    pub train: bool,
    pub render: bool,
}

impl<O, A, OF, AF> Clone for BorderAtariEnvConfig<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    fn clone(&self) -> Self {
        Self {
            rom_dir: self.rom_dir.clone(),
            name: self.name.clone(),
            obs_filter_config: self.obs_filter_config.clone(),
            act_filter_config: self.act_filter_config.clone(),
            train: self.train,
            render: self.render,
        }
    }
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
            train: true,
            render: false,
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

    /// Sets the evaluation flag.
    pub fn eval(mut self) -> Self {
        self.train = false;
        self
    }

    pub fn render(mut self, render: bool) -> Self {
        self.render = render;
        self
    }
}
