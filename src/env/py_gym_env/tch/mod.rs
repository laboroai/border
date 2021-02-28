//! Modules for interaction of [`super::PyGymEnv`] and [`super::vec::PyVevGymEnv`]
//! with agents implemented with tch-rs.
pub mod obs;
pub mod act_d;
pub mod act_c;
pub mod pong;
pub use obs::TchPyGymEnvObsBuffer;
pub use act_d::TchPyGymEnvDiscreteActBuffer;
pub use act_c::TchPyGymEnvContinuousActBuffer;
