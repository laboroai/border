/// Messages that actors (`Sampler`) receive.
pub enum ActorMessage<T> {
    /// Model parameters for the [Agent](border_core::Agent).
    ModelParams(T)
}

/// Messages that the replay buffer receives.
pub enum ReplayBufferMessage<T> {
    /// Call [ReplayBufferBase](border_core::ReplayBufferBase)`::len()`.
    CallLen,

    /// Call [ReplayBufferBase](border_core::ReplayBufferBase)`::update_priority()`.
    CallUpdatePriority(),

    /// Call [ReplayBufferBase](border_core::ReplayBufferBase)`::batch()`.
    Batch(usize),

    /// Call [ReplayBufferBase](border_core::ReplayBufferBase)`::push()`.
    Push(T)
}

/// Messages that the learner ([Agent](border_core::Agent)) receives.
pub enum LearnerMessage<T> {
    /// Current size of samples in the replay buffer.
    Len(usize),

    /// Sample batch used for training.
    Batch(T),
}
