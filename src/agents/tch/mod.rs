pub mod util;

use tch::{Tensor, nn};

pub trait Model {
    fn forward(&self, xs: &Tensor) -> Tensor;

    fn backward_step(&mut self, loss: &Tensor);

    fn get_var_store(&mut self) -> &mut nn::VarStore;
}
