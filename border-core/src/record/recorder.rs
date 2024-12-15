use super::Record;
use crate::{Agent, Env, ReplayBufferBase};
use anyhow::Result;
use std::path::Path;

/// Writes a record to an output destination with [`Recorder::write`].
pub trait Recorder<E, R>
where
    E: Env,
    R: ReplayBufferBase,
{
    /// Write a record to the [`Recorder`].
    fn write(&mut self, record: Record);

    /// Store the record.
    fn store(&mut self, record: Record);

    /// Writes values aggregated from the stored records.
    fn flush(&mut self, step: i64);

    /// Saves the model of the given agent.
    ///
    /// `base` is the base of the path where the model is saved.
    /// For example, at the 100th iteration of training,
    /// `base` may be `100/`.
    #[allow(unused_variables)]
    fn save_model(&self, base: &Path, agent: &Box<dyn Agent<E, R>>) -> Result<()> {
        unimplemented!();
    }
}
