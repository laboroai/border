pub mod adapter;
pub use adapter::{TchObsAdapter, TchActAdapter};
pub mod replay_buffer;
pub use replay_buffer::ReplayBuffer;
pub mod dqn;
pub use dqn::DQN;
pub mod tch;
pub use self::tch::Model;

// extern tch::{Tensor, nn};

// pub trait Model {
//     fn forward(&self, xs: &Tensor) -> Tensor;

//     fn backward_step(&mut self, loss: &Tensor);

//     fn get_var_store(&mut self) -> &mut nn::VarStore;
// }
