//! Interface of neural networks used in RL agents.
pub mod base;
pub mod multihead;
pub mod model1;
pub mod model2;
pub use base::{ModelBase, Model1, Model2};
pub use multihead::StateValueAndDiscreteActProb;
pub use model1::{Model1_1, Model1_2};
pub use model2::Model2_1;
