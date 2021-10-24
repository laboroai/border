//! Configuration of [BorderAtariEnv](super::BorderAtariEnv).
//!
//! If environment variable `ATARI_ROM_DIR` exists, it is used as the directory
//! from which ROM images of the Atari games is loaded.
use std::{env, default::Default};

/// Configuration of [BorderAtariEnv](super::BorderAtariEnv).
pub struct BorderAtariEnvConfig {
    pub(super) rom_dir: String,
    pub(super) name: String,
}

impl Default for BorderAtariEnvConfig {
    fn default() -> Self {
        let rom_dir = if let Ok(var) = env::var("ATARI_ROM_DIR") {
            var
        } else {
            "".to_string()
        };

        Self {
            rom_dir,
            name: "".to_string(),
        }
    }
}

impl BorderAtariEnvConfig {
    /// Sets the name of the game.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}
