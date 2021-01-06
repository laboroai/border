pub mod util;
pub mod replay_buffer;
pub mod dqn;
pub mod ppo;
pub mod model;
use ppo::ppo_discrete;
pub use replay_buffer::{ReplayBuffer, TchBufferableActInfo, TchBufferableObsInfo, Batch, Batch2};
pub use dqn::{DQN, QNetwork};
pub use ppo_discrete::PPODiscrete;

// use std::{path::Path, error::Error};
// use tch::{Tensor, nn};

// pub trait Model {
//     fn forward(&self, xs: &Tensor) -> Tensor;

//     fn backward_step(&mut self, loss: &Tensor);

//     fn get_var_store(&mut self) -> &mut nn::VarStore;

//     fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>>;

//     fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>>;
// }

// pub trait MultiheadModel {
//     fn forward(&self, xs: &Tensor) -> (Tensor, Tensor);

//     fn backward_step(&mut self, loss: &Tensor);

//     fn get_var_store(&mut self) -> &mut nn::VarStore;

//     fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>>;

//     fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>>;
// }
