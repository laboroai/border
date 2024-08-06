//! Parameters of atari environments
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
/// Specifies training or evaluation mode.
#[derive(Clone)]
// TODO: consider to remove this enum
pub enum AtariWrapper {
    /// Training mode
    Train,

    /// Evaluation mode
    Eval,
}
