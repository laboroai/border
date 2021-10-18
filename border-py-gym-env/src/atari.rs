//! Parameters of atari environments
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
/// Specifies training or evaluation mode.
pub enum AtariWrapper {
    /// Training mode
    Train,

    /// Evaluation mode
    Eval,
}
