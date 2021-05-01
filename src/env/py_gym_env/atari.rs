//! Parameters of atari environments

/// Specifies training or evaluation mode.
pub enum AtariWrapper {
    /// Training mode
    Train,

    /// Evaluation mode
    Eval,
}
