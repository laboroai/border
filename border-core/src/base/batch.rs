//! Batch.

/// A batch of transitions for training agents.
///
/// This trait represents a standard transition `(o, a, o', r, is_done)`,
/// where `o` is an observation, `a` is an action, `o'` is an observation
/// after some time steps. Typically, `o'` is for the next step and used as
/// single-step backup. `o'` can also be for the multiple steps after `o` and
/// in this case it is sometimes called n-step backup.
///
/// The type of `o` and `o'` is the associated type `ObsBatch`.
/// The type of `a` is the associated type `ActBatch`.
pub trait StdBatchBase {
    /// A set of observation in a batch.
    type ObsBatch;

    /// A set of observation in a batch.
    type ActBatch;

    /// Unpack the data `(o_t, a_t, o_t+n, r_t, is_terminated_t, is_truncated_t)`.
    ///
    /// Optionally, the return value has sample indices in the replay buffer and
    /// thier weights. Those are used for prioritized experience replay (PER).
    fn unpack(
        self,
    ) -> (
        Self::ObsBatch,
        Self::ActBatch,
        Self::ObsBatch,
        Vec<f32>,
        Vec<i8>,
        Vec<i8>,
        Option<Vec<usize>>,
        Option<Vec<f32>>,
    );

    /// Returns the number of samples in the batch.
    fn len(&self) -> usize;

    /// Returns `o_t`.
    fn obs(&self) -> &Self::ObsBatch;
}
