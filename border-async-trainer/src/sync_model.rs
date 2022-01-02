/// Synchronizes the model of the agent in asynchronous training.
pub trait SyncModel {
    /// Information of the model.
    type ModelInfo: Clone;

    /// Get `ModelInfo`.
    ///
    /// The first element of the return value is the number of optimization steps.
    fn model_info(&self) -> (usize, Self::ModelInfo);

    /// Synchronizes the model.
    fn sync_model(&mut self, model_info: Self::ModelInfo);
}
