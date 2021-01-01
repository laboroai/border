pub mod util;

use std::{path::Path, error::Error};
use tch::{Tensor, nn};

pub trait Model {
    fn forward(&self, xs: &Tensor) -> Tensor;

    fn backward_step(&mut self, loss: &Tensor);

    fn get_var_store(&mut self) -> &mut nn::VarStore;

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>>;

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>>;
}
