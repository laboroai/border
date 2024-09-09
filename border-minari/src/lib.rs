//! Interfaces for Minari datasets.
use border_core::{
    generic_replay_buffer::{BatchBase, SimpleReplayBuffer},
    Act, Env, Obs,
};
pub mod d4rl;

/// Common interface for Minari datasets.
pub trait MinariDataset {
    /// Observation type.
    type Obs: Obs;

    /// Action type.
    type Act: Act;

    /// Observation batch type.
    type ObsBatch: BatchBase;

    /// Action batch type.
    type ActBatch: BatchBase;

    /// Environment type.
    type Env: Env<Obs = Self::Obs, Act = Self::Act>;

    /// Returns the path to the dataset.
    /// 
    /// This method is used in [`Self::load_dataset`].
    fn get_path(&self) -> &str;

    /// Loads dataset.
    fn load_dataset(&self, name: impl AsRef<str>) -> Self;

    /// Creates replay buffer.
    /// 
    /// Episode information will be lost.
    fn create_replay_buffer(&self) -> SimpleReplayBuffer<Self::ObsBatch, Self::ActBatch>;

    /// Recovers the environment.
    fn recover_environment(&self) -> Self::Env;
}
