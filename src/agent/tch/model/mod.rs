//! Interface of neural networks used in RL agents.
pub mod base;
pub mod model1;
pub mod model2;
pub mod multihead;
pub use base::{Model1, Model2, ModelBase, SubModel};
pub use model1::{Model1_1, Model1_2};
pub use model2::Model2_1;
pub use multihead::StateValueAndDiscreteActProb;
