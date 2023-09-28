//! SubBatch, which consists [`StdBatchBase`](`crate::StdBatchBase`).

/// Represents a SubBatch, which consists [`StdBatchBase`](`crate::StdBatchBase`).
pub trait SubBatch: Clone {
    /// Builds a subbatch with a capacity.
    fn new(capacity: usize) -> Self;

    /// Pushes the samples in `data`.
    fn push(&mut self, i: usize, data: Self);

    /// Takes samples in the batch.
    fn sample(&self, ixs: &Vec<usize>) -> Self;

    /// Convert to a vector with each data in batch as an element.
    fn into_vec(self) -> Vec<Self>;

    /// Concat vectors
    fn concat(vec: Vec<Self>) -> Self;
}
