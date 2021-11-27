//! Asynchronous off-policy trainer.
//!
//! # Messages
//! * From Learner ([Agent](border_core::Agent)) to Actor (`Sampler`)
//!   - `ModelParams`
mod base;
mod messages;
mod replay_buffer_proxy;
pub use base::AsyncTrainer;
