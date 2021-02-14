//! Errors in the library.
use thiserror::Error;

/// Errors in the library.
#[derive(Error, Debug)]
pub enum LrrError {
    /// Record key error.
    #[error("Record key error: {0}")]
    RecordKeyError(String),

    /// Record value type error.
    #[error("Record value type error: {0}")]
    RecordValueTypeError(String)
}
